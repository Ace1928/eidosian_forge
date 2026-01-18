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