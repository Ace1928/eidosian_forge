from typing import List, Optional
from tensorflow.core.function import trace_type
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest
def from_concrete_function(concrete_fn, specialized_flat_specs: Optional[List[tensor_spec.TensorSpec]]=None):
    """Generate the Compiler Ir from tf concrete function with TensorSpec.

  Args:
    concrete_fn: returned by using get_concrete_function.
    specialized_flat_specs: specialized flat tf.TensorSpecs for function args.

  Returns:
    Function callable that generate the HLO text.

  Raises:
      ValueError: if concrete_fn is not "compilable" without concrete
      inputs.
  """
    context.ensure_initialized()
    fn_name = concrete_fn.name
    filtered_flat_specs = specialized_flat_specs or list(nest.flatten(concrete_fn.structured_input_signature))
    if not all((s.shape.is_fully_defined() for s in filtered_flat_specs)):
        raise ValueError(f'Only support static input shape but got inputs = {concrete_fn.inputs}')

    def compiler_ir_generator(stage='hlo', device_name=None):
        device_name = maybe_get_device_name(device_name)
        res_bytes = context.context().get_compiler_ir(device_name=device_name, function_name=fn_name, flat_args=filtered_flat_specs, captured_inputs=concrete_fn.captured_inputs, stage=stage)
        if stage in ('hlo_serialized', 'optimized_hlo_serialized', 'optimized_hlo_proto_serialized'):
            return res_bytes
        else:
            return res_bytes.decode('utf-8')
    return compiler_ir_generator