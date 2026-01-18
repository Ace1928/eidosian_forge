import itertools
from typing import Dict, List, Sequence, Union
from torchgen.api import cpp
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import Argument, NativeFunction, SchemaKind, TensorOptionsArguments
from torchgen.utils import FileManager
def format_postrecord_trace(f: NativeFunction) -> str:
    if not should_trace(f):
        return ''
    if f.func.is_out_fn():
        output_names_outplace = [arg.name for arg in f.func.arguments.out]
        output_names_inplace = cpp.return_names(f)
        if output_names_outplace == output_names_inplace:
            outputs = [f'jit::tracer::addOutput(node, {n});' for n in output_names_outplace]
            return POST_RECORD_TRACE.substitute(add_trace_outputs=outputs)
        selection = SELECT.substitute(cond='force_outplace', true='\n'.join((f'jit::tracer::addOutput(node, {n});' for n in output_names_outplace)), false='\n'.join((f'jit::tracer::addOutput(node, {n});' for n in output_names_inplace)))
        return POST_RECORD_TRACE.substitute(add_trace_outputs=selection)
    else:
        output_names = cpp.return_names(f)
        outputs = [f'jit::tracer::addOutput(node, {n});' for n in output_names]
        return POST_RECORD_TRACE.substitute(add_trace_outputs=outputs)