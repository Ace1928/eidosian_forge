from typing import List, Optional
from tensorflow.core.function import trace_type
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest
def compiler_ir_generator(stage='hlo', device_name=None):
    device_name = maybe_get_device_name(device_name)
    res_bytes = context.context().get_compiler_ir(device_name=device_name, function_name=fn_name, flat_args=filtered_flat_specs, captured_inputs=concrete_fn.captured_inputs, stage=stage)
    if stage in ('hlo_serialized', 'optimized_hlo_serialized', 'optimized_hlo_proto_serialized'):
        return res_bytes
    else:
        return res_bytes.decode('utf-8')