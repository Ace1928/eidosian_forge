import itertools
from typing import Dict, List, Sequence, Union
from torchgen.api import cpp
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import Argument, NativeFunction, SchemaKind, TensorOptionsArguments
from torchgen.utils import FileManager
def format_trace_inputs(f: NativeFunction) -> str:

    def dispatch_trace_input(arg: Union[Argument, TensorOptionsArguments]) -> Sequence[str]:
        if isinstance(arg, TensorOptionsArguments):
            name = 'options'
            return [ADD_TRACE_INPUT.substitute(name=name, input='optTypeMetaToScalarType(options.dtype_opt())'), ADD_TRACE_INPUT.substitute(name=name, input='options.layout()'), ADD_TRACE_INPUT.substitute(name=name, input='options.device()'), ADD_TRACE_INPUT.substitute(name=name, input='options.pinned_memory()')]
        else:
            name = arg.name
            if str(arg.type) == 'Tensor?[]':
                return [f'jit::tracer::addInputs(node, "{name}", {name});']
            else:
                return [ADD_TRACE_INPUT.substitute(name=name, input=name)]
    args: List[Union[Argument, TensorOptionsArguments]] = list(f.func.schema_order_arguments())
    if f.func.is_out_fn():
        num_out_args = len(f.func.arguments.out)
        args = args[:-num_out_args]
    trace_inputs = itertools.chain.from_iterable((dispatch_trace_input(arg) for arg in args))
    if f.func.is_out_fn():
        inplace = [ADD_TRACE_INPUT.substitute(name=f.func.arguments.out[i].name, input=f.func.arguments.out[i].name) for i in range(num_out_args)]
        has_tensor_return = any((r.type.is_tensor_like() for r in f.func.returns))
        has_tensor_input_arg = any((a.type.is_tensor_like() for a in f.func.arguments.flat_non_out))
        is_factory_method = f.category_override == 'factory' or (has_tensor_return and (not has_tensor_input_arg))
        if f.func.name.name.base == 'normal':
            is_factory_method = True
        if is_factory_method:
            outplace = [ADD_TRACE_INPUT.substitute(name='out', input='optTypeMetaToScalarType(out.options().dtype_opt())'), ADD_TRACE_INPUT.substitute(name='out', input='out.options().layout()'), ADD_TRACE_INPUT.substitute(name='out', input='out.options().device()'), ADD_TRACE_INPUT.substitute(name='out', input='out.options().pinned_memory()')]
        else:
            outplace = []
        trace_inputs = itertools.chain(trace_inputs, [SELECT.substitute(cond='tracer_state->force_outplace', true='\n'.join(outplace), false='\n'.join(inplace))])
    return '\n'.join(trace_inputs)