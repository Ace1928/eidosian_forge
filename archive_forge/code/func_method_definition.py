import itertools
from typing import Dict, List, Sequence, Union
from torchgen.api import cpp
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import Argument, NativeFunction, SchemaKind, TensorOptionsArguments
from torchgen.utils import FileManager
@with_native_function
def method_definition(f: NativeFunction) -> str:
    assert cpp.name(f.func) not in MANUAL_TRACER
    formals = ', '.join(['c10::DispatchKeySet ks'] + [f'{cpp.argument_type(a, binds='__placeholder__', symint=True).cpp_type()} {a.name}' for a in f.func.schema_order_arguments()])
    return METHOD_DEFINITION.substitute(return_type=cpp.returns_type(f.func.returns, symint=True).cpp_type(), type_wrapper_name=type_wrapper_name(f), formals=formals, type_definition_body=emit_trace_body(f))