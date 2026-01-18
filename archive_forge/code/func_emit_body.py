import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.context import (
from torchgen.model import (
from torchgen.utils import FileManager, mapMaybe
from .context import with_native_function_with_differentiability_info_and_key
from .gen_inplace_or_view_type import (
from .gen_trace_type import (
@with_native_function_with_differentiability_info_and_key
def emit_body(fn: NativeFunctionWithDifferentiabilityInfo, key: str='Default') -> List[str]:
    assert dispatch_strategy(fn) == 'use_derived'
    f = fn.func
    info = fn.info[key] if fn.info else None
    fw_derivatives = fn.fw_derivatives.get(key, []) if fn.fw_derivatives else []
    name = cpp.name(f.func)
    inplace = f.func.kind() == SchemaKind.inplace
    is_out_fn = f.func.kind() == SchemaKind.out
    returns_void = len(f.func.returns) == 0
    base_name = get_base_name(f)
    view_info = get_view_info(f)
    is_foreach = name.startswith('_foreach')
    is_inplace_foreach = is_foreach and inplace
    if is_inplace_foreach:
        inplace_foreacharg2refarg: Dict[Argument, Argument] = {}
        refargname2inplace_foreacharg: Dict[str, Argument] = {}
        base_name_and_overload_name = (f.func.name.name.base, f.func.name.overload_name)
        if info is None:
            assert base_name_and_overload_name in _foreach_ops_without_differentiability_info, f'{'.'.join(base_name_and_overload_name)} should have a differentiability info'
        else:
            assert len(f.func.arguments.flat_non_out) == len(info.func.func.arguments.flat_non_out) or base_name_and_overload_name in _foreach_ops_with_different_arity, f'{'.'.join(base_name_and_overload_name)} has {len(f.func.arguments.flat_non_out)} args but the reference has {len(info.func.func.arguments.flat_non_out)}'
            for foreach_arg, ref_arg in zip(f.func.arguments.flat_non_out, info.func.func.arguments.flat_non_out):
                foreach_arg_type = foreach_arg.type
                if isinstance(foreach_arg_type, ListType):
                    foreach_arg_type = foreach_arg_type.elem
                assert foreach_arg_type == ref_arg.type
                inplace_foreacharg2refarg[foreach_arg] = ref_arg
                refargname2inplace_foreacharg[ref_arg.name] = foreach_arg

    def gen_differentiable_input(arg: Union[Argument, SelfArgument, TensorOptionsArguments]) -> Optional[DifferentiableInput]:
        if isinstance(arg, TensorOptionsArguments):
            return None
        a: Argument = arg.argument if isinstance(arg, SelfArgument) else arg
        cpp_type = cpp.argument_type(a, binds=a.name, symint=True).cpp_type()
        if not is_differentiable(a.name, a.type, info):
            return None
        return DifferentiableInput(name=a.name, type=a.type, cpp_type=cpp_type)

    @with_native_function
    def gen_differentiable_inputs(f: NativeFunction) -> List[DifferentiableInput]:
        arguments = list(f.func.arguments.non_out)
        if is_inplace_foreach and info is not None:
            for i, arg in enumerate(f.func.arguments.flat_non_out):
                if arg in inplace_foreacharg2refarg:
                    mapped_arg = inplace_foreacharg2refarg[arg]
                    arguments[i] = Argument(mapped_arg.name, mapped_arg.type, mapped_arg.default, mapped_arg.annotation)
        return list(mapMaybe(gen_differentiable_input, arguments))

    def find_args_with_derivatives(differentiable_inputs: List[DifferentiableInput]) -> List[DifferentiableInput]:
        """Find arguments that have derivative definitions"""
        if info is None or not info.has_derivatives:
            return differentiable_inputs
        names = {name for d in info.derivatives for name in d.var_names}
        differentiable = [arg for arg in differentiable_inputs if arg.name in names]
        if len(differentiable) != len(names):
            missing = names - {arg.name for arg in differentiable}
            raise RuntimeError(f'Missing arguments for derivatives: {missing} in {info.name}')
        return differentiable
    differentiable_inputs = gen_differentiable_inputs(f)
    args_with_derivatives = find_args_with_derivatives(differentiable_inputs)
    differentiable_outputs = gen_differentiable_outputs(fn, key)
    undifferentiable = base_name in DONT_REQUIRE_DERIVATIVE or name in DONT_REQUIRE_DERIVATIVE
    requires_derivative = not undifferentiable and len(differentiable_inputs) > 0 and (len(differentiable_outputs) > 0 or is_inplace_foreach)
    if info is not None and info.has_derivatives and (not requires_derivative) and (len(f.func.returns) > 0):
        raise RuntimeError(f'ERROR: derivative ignored for {name} -- specified an autograd function without derivative')
    if requires_derivative and len(fw_derivatives) > 0 and (not is_inplace_foreach):
        assert sum((len(derivative.var_names) for derivative in fw_derivatives)) == len(differentiable_outputs), 'Expected the number of forward derivatives implemented to match the number of differentiable outputs. NB: This only applies when at least one forward derivative is implemented. Not implementing any forward derivatives is also okay, and we would require inputs to the op to not have associated tangents in that case.'
    try_jit_decomposition = requires_derivative and len(fw_derivatives) == 0 and (not modifies_arguments(f)) and (not returns_void)

    def emit_save_inputs() -> List[str]:
        setup: List[str] = []
        if info is None or not info.has_derivatives:
            return setup
        has_tensorlist_arg = any((is_tensor_list_type(arg.type) for arg in args_with_derivatives))

        def guard_for(arg: SavedAttribute) -> Optional[str]:
            assert info is not None
            if has_tensorlist_arg and (not is_inplace_foreach):
                return None
            if 'backward' in info.name:
                return None
            if len(args_with_derivatives) <= 1:
                return None
            if arg.nctype.type != BaseCType(tensorT):
                return None
            used_in = [d for d in info.derivatives if arg in d.saved_inputs]
            assert len(used_in) > 0
            if len(used_in) != 1:
                return None
            derivative = used_in[0]
            if len(derivative.var_names) != 1:
                wrap_opt_if_start = derivative.formula.find(f'wrap_opt_if({arg.nctype.name}')
                if wrap_opt_if_start == -1:
                    return None
                wrap_opt_if_match = re.match(f'wrap_opt_if\\({arg.nctype.name},(.*?)\\)', derivative.formula[wrap_opt_if_start:])
                assert wrap_opt_if_match is not None
                condition_slice = slice(len(f'wrap_opt_if\\({arg.nctype.name},'), -1)
                wrap_opt_if_condition = wrap_opt_if_match.group(0)[condition_slice].strip()
                wrap_opt_if_condition = re.sub('grad_input_mask\\[(\\d+)\\]', 'grad_fn->should_compute_output(\\1)', wrap_opt_if_condition)
                return f'{wrap_opt_if_condition}'
            derivative_var_name = derivative.var_names[0]
            for edge_off, a in enumerate(args_with_derivatives):
                if a.name == derivative_var_name:
                    break
            else:
                raise AssertionError()
            return f'grad_fn->should_compute_output({edge_off})'
        if is_inplace_foreach:
            save_input_stmts = save_variables(info.all_saved_inputs, False, guard_for)
            if save_input_stmts:
                setup.append(LOOP_OVER_VECTOR_OF_GRAD_FNS.substitute(preamble='', statements=save_input_stmts))
        else:
            setup.extend(save_variables(info.all_saved_inputs, False, guard_for))
            for arg in args_with_derivatives:
                if is_tensor_list_type(arg.type):
                    setup.append(f'grad_fn->{arg.name}_size_ = {arg.name}.size();')
        return setup

    def setup_derivative(differentiable_inputs: List[DifferentiableInput]) -> List[str]:
        body: List[str] = []
        if is_out_fn:
            body.append(DECLARE_GRAD_FN.substitute(op='Node'))
            body.append(SETUP_NONE_REQUIRES_GRAD.substitute(base_name=base_name, args_to_check=[arg.name for arg in differentiable_inputs]))
            body.append(SETUP_NONE_REQUIRES_GRAD.substitute(base_name=base_name, args_to_check=[arg.name for arg in differentiable_outputs]))
            return body
        op = info.op if info is not None and info.has_derivatives else 'NotImplemented'
        setup = []
        if not is_inplace_foreach:
            setup.extend(ASSIGN_GRAD_FN.substitute(op=op, op_ctor='' if info is not None and info.has_derivatives else f'"{cpp.name(f.func)}"', args_with_derivatives=[arg.name for arg in args_with_derivatives]).split('\n'))
        else:
            list_like_arg = 'self'
            args = [arg.name for arg in args_with_derivatives]
            for i, arg in enumerate(args):
                if is_inplace_foreach and info is not None:
                    if arg in refargname2inplace_foreacharg:
                        foreach_arg = refargname2inplace_foreacharg[arg]
                        args[i] = foreach_arg.name + ('[i]' if isinstance(foreach_arg.type, ListType) else '')
                elif arg == list_like_arg:
                    args[i] = arg + '[i]'
            setup.extend(ASSIGN_VECTOR_OF_GRAD_FN.substitute(op=op, op_ctor='' if info is not None and info.has_derivatives else f'"{cpp.name(f.func)}"', args_with_derivatives=args, irange=f'{list_like_arg}.size()').split('\n'))
        setup.extend(emit_save_inputs())
        body.extend(emit_check_no_requires_grad(differentiable_inputs, args_with_derivatives))
        declare_grad_fn_template = DECLARE_GRAD_FN if not is_inplace_foreach else DECLARE_VECTOR_OF_GRAD_FN
        body.append(declare_grad_fn_template.substitute(op=op))
        body.append(SETUP_DERIVATIVE.substitute(setup=setup))
        return body

    def emit_check_if_in_complex_autograd_allowlist() -> List[str]:
        body: List[str] = []
        if base_name in GRADIENT_IMPLEMENTED_FOR_COMPLEX:
            return body
        for arg in differentiable_outputs:
            name = arg.name
            if arg.cpp_type == 'at::Tensor' or arg.cpp_type in TENSOR_LIST_LIKE_CTYPES:
                body.append(f'throw_error_for_complex_autograd({name}, "{base_name}");')
        return body

    def emit_check_no_requires_grad(tensor_args: List[DifferentiableInput], args_with_derivatives: List[DifferentiableInput]) -> List[str]:
        """Checks that arguments without derivatives don't require grad"""
        body: List[str] = []
        for arg in tensor_args:
            if arg in args_with_derivatives:
                continue
            arg_name = arg.name
            if info and arg_name in info.non_differentiable_arg_names:
                continue
            if arg_name == 'output':
                continue
            body.append(f'check_no_requires_grad({arg_name}, "{arg_name}", "{name}");')
        return body

    def emit_original_self_definition() -> List[str]:
        body: List[str] = []
        if inplace:
            if is_inplace_foreach:
                body.append('std::vector<c10::optional<at::Tensor>> original_selfs(self.size());')
            else:
                body.append('c10::optional<at::Tensor> original_self;')
            all_forward_grad_cond = []
            for derivative in fw_derivatives:
                if derivative.required_original_self_value:
                    all_forward_grad_cond.append(get_any_has_forward_grad_name(derivative.var_names))
            if all_forward_grad_cond:
                if not is_inplace_foreach:
                    body.append(f'if ({' || '.join(all_forward_grad_cond)}) {{')
                    body.append('  original_self = self.clone();')
                    body.append('}')
                else:
                    current_all_forward_grad_cond = [f'{cond}[i]' for cond in all_forward_grad_cond]
                    body.append('for (const auto& i : c10::irange(self.size())) {')
                    body.append(f'  if ({' || '.join(current_all_forward_grad_cond)}) {{')
                    body.append('    original_selfs[i] = self[i].clone();')
                    body.append('  }')
                    body.append('}')
        return body

    def save_variables(saved_variables: Sequence[SavedAttribute], is_output: bool, guard_for: Callable[[SavedAttribute], Optional[str]]=lambda name: None) -> Sequence[str]:
        stmts: List[str] = []
        for arg in sorted(saved_variables, key=lambda sa: str(sa.nctype.name)):
            name = arg.nctype.name.name if isinstance(arg.nctype.name, SpecialArgName) else arg.nctype.name
            foreacharg: Optional[Argument] = None
            is_foreacharg_list_type: bool = False
            type = arg.nctype.type
            expr = arg.expr
            stmts_prepend = None
            if is_inplace_foreach and info is not None:
                name_to_query = name.split('_scalar_type')[0]
                if name_to_query in refargname2inplace_foreacharg:
                    foreacharg = refargname2inplace_foreacharg[name_to_query]
                    is_foreacharg_list_type = isinstance(foreacharg.type, ListType)
                if foreacharg is not None:
                    name_in_expr = f'{foreacharg.name}{('[i]' if is_foreacharg_list_type else '')}'
                    src_name = name
                    if '_scalar_type' in src_name:
                        split_src_name = src_name.split('_scalar_type')
                        assert len(split_src_name) == 2
                        src_name = split_src_name[0]
                    expr = expr.replace(src_name, name_in_expr)
            if type == BaseCType(tensorT) or type == OptionalCType(BaseCType(tensorT)) or type == MutRefCType(OptionalCType(BaseCType(tensorT))) or (is_output and type == BaseCType(scalarT)):
                var = name
                name += '_'
                if var == 'self' and inplace:
                    original_self_var = 'original_self' if not is_inplace_foreach else 'original_selfs[i]'
                    self_var = var if not is_inplace_foreach else var + '[i]'
                    stmts_prepend = f'if (!{original_self_var}.has_value()) {original_self_var} = {self_var}.clone()'
                    var = f'{original_self_var}.value()'
                    assert not is_output
                if inplace and is_output:
                    assert name == 'result_'
                    var = 'self[i]' if is_inplace_foreach or is_foreacharg_list_type else 'self'
                    is_inplace_view = f'{var}.is_view()'
                    expr = f'SavedVariable({var}, {str(is_output).lower()}, {is_inplace_view})'
                else:
                    expr = f'SavedVariable({var}, {str(is_output).lower()})'
                    if foreacharg is not None and 'original_selfs' not in expr:
                        expr = expr.replace(src_name, name_in_expr)
            elif type == BaseCType(tensorListT) or type == ListCType(OptionalCType(BaseCType(tensorT))) or type == BaseCType(iTensorListRefT) or (type == VectorCType(BaseCType(tensorT))):
                if type == VectorCType(BaseCType(tensorT)):
                    assert is_foreach and is_output
                expr = f'make_saved_variable_list({name}, {str(is_foreach and is_output).lower()})'
                name += '_'
            elif type == BaseCType(intArrayRefT):
                expr = expr + '.vec()'
            elif type == BaseCType(symIntArrayRefT):
                expr = expr + '.vec()'
            elif type == BaseCType(stringT):
                expr = f'std::string({expr})'
            elif type == OptionalCType(BaseCType(stringT)):
                expr = f'{expr}.has_value() ? c10::optional<std::string>(std::string({expr}.value())) : c10::nullopt'
            elif type == ArrayRefCType(elem=BaseCType(type=BaseCppType(ns='at', name='Scalar'))):
                expr = expr + '.vec()'
            guard = guard_for(arg)
            if guard is None:
                if stmts_prepend:
                    stmts.append(f'{stmts_prepend};')
                stmts.append(f'grad_fn->{name} = {expr};')
            else:
                stmts.append(f'if ({guard}) {{')
                if stmts_prepend:
                    stmts.append(f'  {stmts_prepend};')
                stmts.append(f'  grad_fn->{name} = {expr};')
                stmts.append('}')
        return stmts

    def emit_dispatch_call(f: NativeFunction, input_base: str, unpacked_args: Sequence[str]) -> str:
        """Dispatch call via function in a namespace or method on Tensor."""
        dispatcher_sig = DispatcherSignature.from_schema(f.func)
        dispatcher_exprs = dispatcher_sig.exprs()
        dispatch_key_set = 'ks & c10::after_autograd_keyset'
        call = CALL_REDISPATCH.substitute(api_name=cpp.name(f.func, faithful_name_for_out_overloads=True, symint_overload=f.func.has_symint()), unpacked_args=[dispatch_key_set] + list(unpacked_args))
        return call

    def wrap_output(f: NativeFunction, unpacked_bindings: List[Binding], var: str) -> str:
        call = ''
        rhs_value: Optional[str] = None
        if not any((r.type.is_tensor_like() for r in f.func.returns)):
            rhs_value = var
        else:
            rhs_value = f'std::move({var})'
        assert rhs_value is not None
        call += ASSIGN_RETURN_VALUE.substitute(return_values=tie_return_values(f), rhs_value=rhs_value)
        return call

    def check_tensorimpl_and_storage(call: str, unpacked_bindings: List[Binding]) -> str:
        stmts_before_call: List[str] = []
        stmts_after_call: List[str] = []
        if cpp.name(f.func) in DONT_ENFORCE_SAME_TENSOR_IMPL_OR_STORAGE:
            return call
        for unpacked_binding in unpacked_bindings:
            arg = unpacked_binding.name
            noref_cpp_type = unpacked_binding.nctype.type.remove_const_ref()
            if noref_cpp_type == BaseCType(tensorListT) or noref_cpp_type == BaseCType(iTensorListRefT):
                stmts_before_call += [SAVE_TENSORLIST_STORAGE.substitute(tensorlist_name=arg), SAVE_TENSORLIST_IMPL.substitute(tensorlist_name=arg)]
                stmts_after_call += [ENFORCE_SAME_TENSORLIST_STORAGE.substitute(tensorlist_name=arg), ENFORCE_SAME_TENSORLIST_IMPL.substitute(tensorlist_name=arg)]
            elif noref_cpp_type == ListCType(OptionalCType(BaseCType(tensorT))):
                stmts_before_call += [SAVE_OPTIONALTENSORLIST_STORAGE.substitute(tensorlist_name=arg), SAVE_OPTIONALTENSORLIST_IMPL.substitute(tensorlist_name=arg)]
                stmts_after_call += [ENFORCE_SAME_OPTIONALTENSORLIST_STORAGE.substitute(tensorlist_name=arg), ENFORCE_SAME_OPTIONALTENSORLIST_IMPL.substitute(tensorlist_name=arg)]
            elif noref_cpp_type == BaseCType(tensorT):
                stmts_before_call += [SAVE_TENSOR_STORAGE.substitute(tensor_name=arg), SAVE_TENSOR_IMPL.substitute(tensor_name=arg)]
                stmts_after_call += [ENFORCE_SAME_TENSOR_STORAGE.substitute(tensor_name=arg, out_tensor_name=arg), ENFORCE_SAME_TENSOR_IMPL.substitute(tensor_name=arg)]
        assert stmts_before_call and stmts_after_call or (not stmts_before_call and (not stmts_after_call))
        if f.func.kind() not in (SchemaKind.inplace, SchemaKind.out):
            base_name = f.func.name.name.base
            aliased_arg_name = ALL_VIEW_FUNCTIONS.get(base_name, None)
            if aliased_arg_name is not None:
                aliased_arg_name = unpacked_name(aliased_arg_name)
            for i, (ret, ret_name) in enumerate(zip(f.func.returns, cpp.return_names(f))):
                noref_cpp_type = cpp.return_type(ret, symint=True).remove_const_ref()
                if noref_cpp_type == BaseCType(tensorT):
                    if aliased_arg_name is not None:
                        assert i == 0, 'Expect non-CompositeImplicitAutograd view function {base} to return single output'
                        stmts_after_call += [ENFORCE_SAME_TENSOR_STORAGE.substitute(tensor_name=aliased_arg_name, out_tensor_name=ret_name)]
                    elif type_wrapper_name(f) not in DONT_ENFORCE_STORAGE_IMPL_USE_COUNT:
                        stmts_after_call += [ENFORCE_TENSOR_STORAGE_USE_COUNT_EQUALS_ONE.substitute(tensor_name=ret_name, fn_name=type_wrapper_name(f))]
                    if type_wrapper_name(f) not in DONT_ENFORCE_TENSOR_IMPL_USE_COUNT:
                        stmts_after_call += [ENFORCE_TENSOR_IMPL_USE_COUNT_LT_OR_EQ_ONE.substitute(tensor_name=ret_name, fn_name=type_wrapper_name(f))]
                elif noref_cpp_type == ListCType(OptionalCType(BaseCType(tensorT))):
                    raise AssertionError(f'Please add use_count checks for {noref_cpp_type}')
                elif noref_cpp_type == BaseCType(tensorListT):
                    raise AssertionError(f'Please add use_count checks for {noref_cpp_type}')
        if stmts_before_call and stmts_after_call:
            call = RUN_ONLY_IN_DEBUG_MODE.substitute(statements=stmts_before_call) + call + RUN_ONLY_IN_DEBUG_MODE.substitute(statements=stmts_after_call)
        return call

    def emit_call(f: NativeFunction, unpacked_bindings: List[Binding], try_jit_decomposition: bool) -> str:
        unpacked_args = [b.name for b in unpacked_bindings]
        base_type_call = emit_dispatch_call(f, 'self_', unpacked_args)
        if get_view_info(f) is not None or modifies_arguments(f):
            guard = 'at::AutoDispatchBelowAutograd guard;'
        else:
            guard = 'at::AutoDispatchBelowADInplaceOrView guard;'
        any_has_forward_grad = get_any_has_fw_grad_cond(derivative=None) if requires_derivative else 'false'
        return_types = ', '.join([cpp.return_type(a, symint=True).cpp_type() for a in f.func.returns])
        if len(f.func.returns) > 1:
            return_types = f'std::tuple<{return_types}>'
        arg_names = [a.name for a in cpp.arguments(f.func.arguments, faithful=True, symint=True, method=False, cpp_no_default_args=set())]
        if not modifies_arguments(f) and (not returns_void):
            if try_jit_decomposition:
                call = DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES_JVP_DECOMP.substitute(base_type_call=base_type_call, tmp_var=TMP_VAR, guard=guard, any_has_forward_grad=any_has_forward_grad, op_name=cpp.name(f.func), op_overload=f.func.name.overload_name, return_types=return_types, arg_names=arg_names)
            else:
                call = DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES.substitute(base_type_call=base_type_call, tmp_var=TMP_VAR, guard=guard)
            call += wrap_output(f, unpacked_bindings, TMP_VAR)
        else:
            assert not try_jit_decomposition
            call = DISPATCH_TO_NON_VAR_TYPE_WITHOUT_RETURN_VALUES.substitute(base_type_call=base_type_call, guard=guard)
        call = check_tensorimpl_and_storage(call, unpacked_bindings)
        return call

    def emit_history() -> str:
        fn = 'rebase' if modifies_arguments(f) and view_info is None else 'set'
        output_names = [r.name for r in differentiable_outputs]
        outs = CodeTemplate('flatten_tensor_args( ${outs} )').substitute(outs=output_names if not is_inplace_foreach else 'self')
        if not is_inplace_foreach:
            return SET_HISTORY.substitute(fn=fn, differentiable_outputs=outs)
        else:
            return LOOP_OVER_VECTOR_OF_GRAD_FNS.substitute(preamble=f'auto differentiable_outputs = {outs};\nTORCH_INTERNAL_ASSERT(differentiable_outputs.size() == grad_fns.size());', statements=f'{fn}_history(differentiable_outputs[i], grad_fns[i]);')

    def emit_save_outputs() -> str:
        if is_out_fn:
            return ''
        if info is not None and info.has_derivatives:
            stmts = save_variables(info.all_saved_outputs, True)
            if len(stmts) == 0:
                return ''
            if not is_inplace_foreach:
                return CONDITIONAL.substitute(cond='grad_fn', statements=stmts)
            else:
                return LOOP_OVER_VECTOR_OF_GRAD_FNS.substitute(preamble='', statements=stmts)
        return ''

    def emit_any_requires_grad() -> List[str]:
        extra_condition = ''
        if info and info.output_differentiability_conditions:
            assert len(info.output_differentiability_conditions) == 1
            extra_condition = f'_any_requires_grad &= ({info.output_differentiability_conditions[0]});'
        names_of_args_with_derivatives = [arg.name for arg in args_with_derivatives]
        if is_inplace_foreach and info is not None:
            for i, arg in enumerate(names_of_args_with_derivatives):
                for f_arg, r_arg in inplace_foreacharg2refarg.items():
                    if arg == r_arg.name:
                        names_of_args_with_derivatives[i] = f_arg.name
        return [SETUP_ANY_REQUIRES_GRAD.substitute(args_with_derivatives=names_of_args_with_derivatives, extra_differentiability_conditions=extra_condition)]

    def get_any_has_forward_grad_name(var_names: Tuple[str, ...]) -> str:
        if len(var_names) == 1:
            return f'_any_has_forward_grad_{var_names[0]}'
        else:
            return f'_any_has_forward_grad_{'_'.join(var_names)}'

    def emit_any_has_forward_grad() -> List[str]:
        content: List[str] = []
        if not is_foreach:
            for derivative in fw_derivatives:
                requires_fw_grad = get_any_has_fw_grad_cond(derivative=derivative)
                if info and info.output_differentiability_conditions:
                    assert len(info.output_differentiability_conditions) == 1
                    requires_fw_grad = f'({info.output_differentiability_conditions[0]}) && {requires_fw_grad}'
                content.append(f'[[maybe_unused]] auto {get_any_has_forward_grad_name(derivative.var_names)} = {requires_fw_grad};')
        else:
            for derivative in fw_derivatives:
                bool_vector_name = get_any_has_forward_grad_name(derivative.var_names)
                cur_derivative_conditions = []
                for inp in differentiable_inputs:
                    if derivative.required_inputs_fw_grad is None:
                        continue
                    if inp.name not in derivative.required_inputs_fw_grad:
                        continue
                    inp_name = inp.name if not inplace else refargname2inplace_foreacharg[inp.name].name
                    inp_type = inp.type if not inplace else refargname2inplace_foreacharg[inp.name].type
                    is_list_type = is_tensor_list_type(inp_type)
                    if is_list_type:
                        if inp_name != 'self':
                            content.append(FW_DERIVATIVE_SIZE_CHECK_TEMPLATE.substitute(inp_name=inp_name))
                        cur_derivative_conditions.append(FW_DERIVATIVE_CHECK_TEMPLATE.substitute(req_inp=inp_name + '[i]'))
                    else:
                        cur_derivative_conditions.append(FW_DERIVATIVE_CHECK_TEMPLATE.substitute(req_inp=inp_name))
                content.append(f'std::vector<bool> {bool_vector_name}(self.size());')
                content.append('for (const auto& i : c10::irange(self.size())) {')
                content.append(f'  {bool_vector_name}[i] = {' || '.join(cur_derivative_conditions)};')
                content.append('}')
        return content

    def emit_check_inplace() -> List[str]:
        if not inplace:
            return []
        return [f'check_inplace({arg.name}, _any_requires_grad);' for arg in differentiable_outputs]

    def emit_fw_derivatives() -> List[str]:
        content: List[str] = []
        fw_grad_setters: List[str] = []
        for derivative in fw_derivatives:
            res = derivative.var_names
            if f.func.name.name.inplace:
                assert len(res) == 1, 'Expected number of outputs to be 1 if function is inplace'
                res = ('self',)
            assert derivative.required_inputs_fw_grad is not None
            unpacked_arguments = ''
            for inp in differentiable_inputs:
                inp_name = inp.name
                is_input_tensorlist = is_foreach and is_tensor_list_type(inp.type if not inplace else refargname2inplace_foreacharg[inp.name].type)
                input_suffix = '[i]' if is_input_tensorlist else ''
                if is_inplace_foreach:
                    if inp.name in refargname2inplace_foreacharg:
                        inp_name = refargname2inplace_foreacharg[inp.name].name
                zeros_fn = 'zeros' if inplace and inp.name == 'self' else '_efficientzerotensor'
                if inp.name in derivative.required_inputs_fw_grad:
                    unpacked_arguments += FW_DERIVATIVE_DEFINED_GRAD_TEMPLATE.substitute(inp_name=inp.name, inp=inp_name + input_suffix, zeros_fn=zeros_fn)
                if inp.name in (derivative.required_inputs_primal or []):
                    unpacked_arguments += FW_DERIVATIVE_DEFINED_PRIMAL_TEMPLATE.substitute(inp_name=inp.name, inp=inp_name + input_suffix)
            if derivative.required_original_self_value:
                input_suffix = 's[i]' if is_inplace_foreach else ''
                unpacked_arguments += FW_DERIVATIVE_DEFINED_GRAD_TEMPLATE.substitute(inp_name='original_self', inp='original_self' + input_suffix, zeros_fn=zeros_fn)
                unpacked_arguments += FW_DERIVATIVE_DEFINED_PRIMAL_TEMPLATE.substitute(inp_name='original_self', inp='original_self' + input_suffix)
            elif inplace and derivative.is_reusing_outplace_formula:
                unpacked_arguments += 'self_t = GradMode::is_enabled() ? self_t.clone() : self_t;'
            if inplace:
                is_inplace_str = 'true'
            else:
                is_inplace_str = 'false'
            requires_fw_grad = get_any_has_forward_grad_name(derivative.var_names)
            if all((isinstance(var_type, BaseType) and var_type.is_tensor_like() for var_type in derivative.var_types)):
                if len(derivative.var_types) == 1:
                    opt_res_grad_type = OptionalCType(BaseCType(tensorT)).cpp_type()
                    if not is_foreach:
                        fw_grad_setters.append(FW_DERIVATIVE_SETTER_TENSOR.substitute(out_arg=res[0], is_inplace=is_inplace_str))
                    else:
                        assert res[0] == ('result' if not inplace else 'self')
                        fw_grad_setters.append(FW_DERIVATIVE_SETTER_TENSOR_FOREACH.substitute(out_arg=res[0], is_inplace=is_inplace_str))
                    requires_fw_grad += f' && ({derivative.var_names[0]}.defined())'
                else:
                    tuple_type = TupleCType([BaseCType(tensorT)] * len(derivative.var_types))
                    opt_res_grad_type = OptionalCType(tuple_type).cpp_type()
                    for idx, single_res in enumerate(res):
                        fw_grad_setters.append(FW_DERIVATIVE_SETTER_MULTI_OUTPUT.substitute(idx=idx, all_res='_'.join(res), out_arg=single_res))
            elif isinstance(derivative.var_types[0], ListType) and derivative.var_types[0].is_tensor_like():
                assert len(derivative.var_types) == 1, 'Expected number of outputs to be 1 if function returns ListType'
                if not is_foreach:
                    opt_res_grad_type = OptionalCType(VectorCType(BaseCType(tensorT))).cpp_type()
                    fw_grad_setters.append(FW_DERIVATIVE_SETTER_TENSOR_LIST.substitute(out_arg=res[0], is_inplace=is_inplace_str))
                else:
                    opt_res_grad_type = OptionalCType(BaseCType(tensorT)).cpp_type()
                    fw_grad_setters.append(FW_DERIVATIVE_SETTER_TENSOR_FOREACH.substitute(out_arg=res[0], is_inplace=is_inplace_str))
            else:
                raise RuntimeError('Unsupported output type for forward derivative')
            if not is_foreach:
                fw_grad_opt_definition = f'{opt_res_grad_type} {'_'.join(res)}_new_fw_grad_opt = c10::nullopt;'
                content.append(FW_DERIVATIVE_TEMPLATE.substitute(fw_grad_opt_definition=fw_grad_opt_definition, requires_fw_grad=requires_fw_grad, formula=derivative.formula, out_arg='_'.join(res), unpacked_arguments=unpacked_arguments))
            else:
                fw_grad_opt_definition = f'std::vector<{opt_res_grad_type}> {'_'.join(res)}_new_fw_grad_opts(self.size(), c10::nullopt);'
                foreach_forward_grad_formula = derivative.formula
                _foreach_arg: Union[Argument, DifferentiableInput]
                if inplace:
                    for _foreach_arg, _ref_arg in inplace_foreacharg2refarg.items():
                        if not (is_tensor_type(_foreach_arg.type) or is_tensor_list_type(_foreach_arg.type)):
                            pattern = _foreach_arg.name
                            if isinstance(_foreach_arg.type, ListType):
                                pattern += '[i]'
                            foreach_forward_grad_formula = foreach_forward_grad_formula.replace(_ref_arg.name, pattern)
                elif 'result' in foreach_forward_grad_formula and 'result[i]' not in foreach_forward_grad_formula:
                    foreach_forward_grad_formula = foreach_forward_grad_formula.replace('result', 'result[i]')
                content.append(FW_DERIVATIVE_FOREACH_TEMPLATE.substitute(fw_grad_opt_definition=fw_grad_opt_definition, vector_of_optional_tensor=f'{'_'.join(res)}_new_fw_grad_opts', any_has_forward_grad_for_current_index=' || '.join((get_any_has_forward_grad_name(derivative.var_names) + '[i]' for derivative in fw_derivatives)), formula=foreach_forward_grad_formula, unpacked_arguments=unpacked_arguments))
        content.append('\n'.join(fw_grad_setters))
        return content

    def get_any_has_fw_grad_cond(derivative: Optional[ForwardDerivative]) -> str:
        if derivative is None:
            to_check: List[str] = []
            for inp in list(mapMaybe(gen_differentiable_input, f.func.arguments.non_out + list(f.func.arguments.out))):
                if is_tensor_type(inp.type):
                    to_check.append(FW_DERIVATIVE_CHECK_TEMPLATE.substitute(req_inp=inp.name))
                elif is_tensor_list_type(inp.type):
                    to_check.append(FW_DERIVATIVE_TENSORLIST_CHECK_TEMPLATE.substitute(req_inp=inp.name))
                else:
                    raise RuntimeError(f'Unsupported input type for "{name}" when forbidding forward AD usage.')
            return f'({' || '.join(to_check)})'
        else:
            assert derivative.required_inputs_fw_grad is not None
            if len(derivative.required_inputs_fw_grad) == 0:
                if not (len(differentiable_inputs) == 1 and is_tensor_list_type(differentiable_inputs[0].type)):
                    raise RuntimeError(f'No differentiable input to "{name}" is a differentiable Tensor (as the provided forward AD formula does not use any input tangent) even though a forward gradient formula has been defined for it. This case should only happen for function that take a single TensorList as input. All other cases are not supported right now.')
                any_has_fw_grad = 'true'
            else:
                any_has_fw_grad = ' || '.join([(FW_DERIVATIVE_TENSORLIST_CHECK_TEMPLATE if is_tensor_list_type(inp.type) else FW_DERIVATIVE_CHECK_TEMPLATE).substitute(req_inp=inp.name) for inp in differentiable_inputs if inp.name in derivative.required_inputs_fw_grad])
                any_has_fw_grad = f'({any_has_fw_grad})'
            return any_has_fw_grad

    def emit_forbid_fw_derivatives(is_out_fn: bool=False) -> str:
        if is_out_fn:
            msg = 'because it is an out= function'
        else:
            msg = 'because it has not been implemented yet.\\nPlease file an issue to PyTorch at https://github.com/pytorch/pytorch/issues/new?template=feature-request.yml so that we can prioritize its implementation.'
        cond = get_any_has_fw_grad_cond(derivative=None)
        return FW_DERIVATIVE_FORBID_TEMPLATE.substitute(cond=cond, name=name, msg=msg) if cond != '' else ''
    body: List[str] = []
    unpack_args_stats, unpacked_bindings = unpack_args(f)
    body.extend(unpack_args_stats)
    if requires_derivative:
        body.extend(emit_any_requires_grad())
        body.extend(emit_any_has_forward_grad())
        body.extend(emit_check_inplace())
        body.extend(emit_original_self_definition())
        body.extend(setup_derivative(differentiable_inputs))
    body.append(declare_returned_variables(f))
    body.append(emit_call(f, unpacked_bindings, try_jit_decomposition))
    if requires_derivative:
        body.append(emit_history())
        body.extend(emit_check_if_in_complex_autograd_allowlist())
    if is_out_fn:
        body.append(emit_forbid_fw_derivatives(is_out_fn=True))
    elif requires_derivative and (not try_jit_decomposition):
        if len(fw_derivatives) > 0:
            body.extend(emit_fw_derivatives())
        else:
            body.append(emit_forbid_fw_derivatives())
    if requires_derivative:
        body.append(emit_save_outputs())
    if str(f.func.name.name) in RESET_GRAD_ACCUMULATOR:
        assert inplace
        body.append('reset_grad_accumulator(self);')
    if not returns_void:
        body.append(f'return {get_return_value(f)};')
    return body