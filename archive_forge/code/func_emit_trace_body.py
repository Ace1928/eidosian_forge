import itertools
from typing import Dict, List, Sequence, Union
from torchgen.api import cpp
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import Argument, NativeFunction, SchemaKind, TensorOptionsArguments
from torchgen.utils import FileManager
def emit_trace_body(f: NativeFunction) -> List[str]:
    trace_body: List[str] = []
    trace_body.append(format_prerecord_trace(f))
    trace_body.append(declare_returned_variables(f))
    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    dispatcher_exprs = dispatcher_sig.exprs()
    dispatch_key_set = 'ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer)'
    redispatch_args = ', '.join([dispatch_key_set] + [a.expr for a in dispatcher_exprs])
    assign_return_values = f'{tie_return_values(f)} = ' if f.func.kind() in [SchemaKind.functional, SchemaKind.mutable] and f.func.returns else ''
    trace_body.append(TRACE_DISPATCH.substitute(assign_return_values=assign_return_values, unambiguous_name=f.func.name.unambiguous_name(), unpacked_args=redispatch_args))
    trace_body.append(format_postrecord_trace(f))
    if f.func.returns:
        trace_body.append(f'return {get_return_value(f)};')
    return trace_body