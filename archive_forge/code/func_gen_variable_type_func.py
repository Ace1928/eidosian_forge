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
def gen_variable_type_func(fn: NativeFunctionWithDifferentiabilityInfo) -> Dict[str, List[str]]:
    f = fn.func
    result = {}
    with native_function_manager(f):
        name = cpp.name(f.func)
        formals = gen_formals(f)
        if fn.info is None and str(f.func.name.name) not in RESET_GRAD_ACCUMULATOR and (get_base_name(f) not in DONT_REQUIRE_DERIVATIVE) and (len(gen_differentiable_outputs(fn)) > 0) and (cpp.name(f.func) not in DONT_ENFORCE_SAME_TENSOR_IMPL_OR_STORAGE) and (type_wrapper_name(f) not in DONT_ENFORCE_STORAGE_IMPL_USE_COUNT) and (type_wrapper_name(f) not in DONT_ENFORCE_TENSOR_IMPL_USE_COUNT):
            type_definition = ''
            wrapper_registration = AUTOGRAD_NOT_IMPLEMENTED_REGISTRATION.substitute(unqual_operator_name_with_overload=f.func.name)
            result['type_derived_method_definitions_Default'] = [type_definition]
            result['wrapper_registrations_Default'] = [wrapper_registration]
        elif not fn.info:
            key = 'Default'
            type_definition = METHOD_DEFINITION.substitute(return_type=cpp.returns_type(f.func.returns, symint=True).cpp_type(), type_wrapper_name=type_wrapper_name(f, key), type_definition_body=emit_body(fn, key), formals=formals)
            wrapper_registration = gen_wrapper_registration(f, key)
            result[f'type_derived_method_definitions_{key}'] = [type_definition]
            result[f'wrapper_registrations_{key}'] = [wrapper_registration]
        else:
            for key in fn.info.keys():
                type_definition = METHOD_DEFINITION.substitute(return_type=cpp.returns_type(f.func.returns, symint=True).cpp_type(), type_wrapper_name=type_wrapper_name(f, key), type_definition_body=emit_body(fn, key), formals=formals)
                wrapper_registration = gen_wrapper_registration(f, key)
                result[f'type_derived_method_definitions_{key}'] = [type_definition]
                result[f'wrapper_registrations_{key}'] = [wrapper_registration]
    assert (name in MANUAL_BACKEND) == f.manual_kernel_registration
    if name in MANUAL_AUTOGRAD_AND_TRACER or (fn.info and any((info.has_derivatives for info in fn.info.values()))):
        msg = f"There's a formula for {name}(or its functional variant) in derivatives.yaml. It's required to add a dispatch section for it with explicit supported backends e.g CPU/CUDA or CompositeExplicitAutograd in native_functions.yaml. Please see https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native#choosing-the-right-dispatch-keyword for instructions to choose the right dispatch keyword."
        assert f.is_abstract, msg
    return result