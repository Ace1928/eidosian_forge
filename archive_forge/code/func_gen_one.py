import itertools
import textwrap
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
import torchgen.api.cpp as cpp
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function, native_function_manager
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import assert_never, mapMaybe, Target
@method_with_native_function
def gen_one(self, f: NativeFunction) -> Optional[str]:
    assert not f.manual_kernel_registration
    if self.target is Target.REGISTRATION and (not self.selector.is_native_function_selected(f)):
        return None
    if self.backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional and f.func.kind() is SchemaKind.out:
        return None
    cpp_sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
    kern = self.backend_index.get_kernel(f)
    sig = NativeSignature(f.func, prefix=f'wrapper_{self.backend_index.dispatch_key}_', symint=kern is not None and kern.supports_symint())
    if self.target is Target.NAMESPACED_DECLARATION:
        result = ''
        for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
            result += f'TORCH_API {cpp_sig.decl()};\n'
        return result
    elif self.target is Target.NAMESPACED_DEFINITION:

        def generate_defn(cpp_sig: CppSignature) -> str:
            return f'\n{cpp_sig.defn()} {{\nreturn {sig.name()}({', '.join((e.expr for e in translate(cpp_sig.arguments(), sig.arguments())))});\n}}\n'
        result = ''
        for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
            result += generate_defn(cpp_sig)
        return result
    elif self.target is Target.ANONYMOUS_DEFINITION:
        k = f.func.kind()
        sig_body = []
        context: List[Union[Binding, Expr]] = list(sig.arguments())
        if self.backend_index.dispatch_key is DispatchKey.Meta:
            class_name = f'structured_{meta.name(self.g)}_meta_{k.name}'
            parent_class = f'at::meta::structured_{meta.name(self.g)}'
        elif self.backend_index.dispatch_key is DispatchKey.CompositeExplicitAutogradNonFunctional:
            class_name = f'structured_{meta.name(self.g)}_default_backend_{k.name}'
            parent_class = f'at::meta::structured_{meta.name(self.g)}'
        else:
            metadata = self.backend_index.get_kernel(self.g)
            assert metadata is not None
            class_name = f'structured_{metadata.kernel}_{k.name}'
            parent_class = f'{metadata.cpp_namespace}::structured_{metadata.kernel}'
        if self.backend_index.device_guard:
            device_check_args = itertools.chain(f.func.arguments.out, f.func.arguments.flat_positional)
            sig_body.append(RegisterDispatchKey.gen_device_check(f.device_check, list(device_check_args), sig.name()))
        if k is SchemaKind.functional:
            sig_body.append(f'{class_name} op;')
        elif k is SchemaKind.inplace:
            sig_body.append(f'{class_name} op(self);')
        elif k is SchemaKind.out:
            out_args_str = ', '.join((a.name for a in f.func.arguments.out))
            sig_body.append(f'{class_name} op({out_args_str});')
        meta_exprs = ', '.join((e.expr for e in translate(context, structured.meta_arguments(self.g), method=False)))
        if self.g.out.precomputed:
            sig_body.append(f'auto precompute = op.meta({meta_exprs});')
            precomputed_values = [*self.g.out.precomputed.replace.values(), self.g.out.precomputed.add]
            for precomputed_elems in precomputed_values:
                for arg in precomputed_elems:
                    context.append(Expr(expr=f'precompute.{arg.name}', type=structured.argument_type(arg, binds=arg.name)))
            sig_body.append('(void)precompute;')
        else:
            sig_body.append(f'op.meta({meta_exprs});')
        out_args = structured.out_arguments(self.g)
        for i, out_arg in enumerate(out_args):
            assert ConstRefCType(BaseCType(tensorT)) == out_arg.nctype.type
            if k is SchemaKind.out:
                expr = f'op.maybe_get_output({i})'
            else:
                expr = f'op.outputs_[{i}]'
            context.append(Expr(expr=expr, type=NamedCType(out_arg.nctype.name, MutRefCType(BaseCType(tensorT)))))
        if self.backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
            out_sig_group = CppSignatureGroup.from_native_function(self.g.out, method=False, fallback_binding=f.manual_cpp_binding)
            out_sig = out_sig_group.most_faithful_signature()
            api_name = out_sig.name()
            out_exprs = ', '.join((e.expr for e in translate(context, out_sig.arguments(), method=False)))
            sig_body.append(f'at::{api_name}({out_exprs});')
        elif self.backend_index.dispatch_key != DispatchKey.Meta:
            impl_exprs = ', '.join((e.expr for e in translate(context, structured.impl_arguments(self.g), method=False)))
            sig_body.append(f'op.impl({impl_exprs});')
        if k is SchemaKind.out or k is SchemaKind.inplace:
            for i in range(len(f.func.returns)):
                sig_body.append(f'if (op.proxy_outputs_[{i}].has_value()) op.outputs_[{i}].get().copy_(*op.proxy_outputs_[{i}]);')
        if k is SchemaKind.functional:
            if len(f.func.returns) == 1:
                ret_expr = 'std::move(op.outputs_[0])'
            else:
                moved = ', '.join((f'std::move(op.outputs_[{i}])' for i in range(len(f.func.returns))))
                ret_expr = f'std::make_tuple({moved})'
        elif k is SchemaKind.inplace:
            ret_expr = 'self'
        elif k is SchemaKind.out:
            if len(f.func.returns) == 1:
                ret_expr = f.func.arguments.out[0].name
            else:
                refs = ', '.join((a.name for a in f.func.arguments.out))
                ret_expr = f'std::forward_as_tuple({refs})'
        sig_body.append(f'return {ret_expr};')
        sig_body_str = '\n'.join(sig_body)
        return f'{self.gen_class(f, k, class_name=class_name, parent_class=parent_class, generate_super=self.g.out.structured_inherits is not None)}\n\n{sig.defn()} {{\n{sig_body_str}\n}}\n'
    elif self.target is Target.REGISTRATION:
        return f'm.impl("{f.func.name}", TORCH_FN({sig.name()}));'
    else:
        assert_never(self.target)
        return None