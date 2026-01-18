import itertools
from typing import Dict, List, Sequence, Union
from torchgen.api import cpp
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import Argument, NativeFunction, SchemaKind, TensorOptionsArguments
from torchgen.utils import FileManager
def format_trace_op_name(f: NativeFunction) -> str:
    if f.func.kind() in (SchemaKind.functional, SchemaKind.out) or f.func.name.name.dunder_method:
        trace_name = str(f.func.name.name)
        trace_name = RENAME_TRACE.get(trace_name, trace_name)
        return OP_NAME.substitute(trace_name=trace_name)
    outplace_trace_name = f.func.name.name.base
    inplace_trace_name = cpp.name(f.func)
    outplace_trace_name = RENAME_TRACE.get(outplace_trace_name, outplace_trace_name)
    inplace_trace_name = RENAME_TRACE.get(inplace_trace_name, inplace_trace_name)
    return SELECT.substitute(cond='tracer_state->force_outplace', true=OP_NAME.substitute(trace_name=outplace_trace_name), false=OP_NAME.substitute(trace_name=inplace_trace_name))