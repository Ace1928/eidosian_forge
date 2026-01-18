import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.util import nest
def arg_is_blockwise(block_dimensions, arg, arg_split_dim):
    """Detect if input should be interpreted as a list of blocks."""
    if isinstance(arg, (tuple, list)) and len(arg) == len(block_dimensions):
        if not any((nest.is_nested(x) for x in arg)):
            return True
        else:
            arg_dims = [tensor_conversion.convert_to_tensor_v2_with_dispatch(x).shape[arg_split_dim] for x in arg]
            self_dims = [dim.value for dim in block_dimensions]
            if all((self_d is None for self_d in self_dims)):
                if len(arg_dims) == 1:
                    return False
                elif any((dim != arg_dims[0] for dim in arg_dims)):
                    return True
                else:
                    raise ValueError('Parsing of the input structure is ambiguous. Please input a blockwise iterable of `Tensor`s or a single `Tensor`.')
            if all((self_d == arg_d or self_d is None for self_d, arg_d in zip(self_dims, arg_dims))):
                return True
            self_dim = sum((self_d for self_d in self_dims if self_d is not None))
            if all((s == arg_dims[0] for s in arg_dims)) and arg_dims[0] >= self_dim:
                return False
            raise ValueError('Input dimension does not match operator dimension.')
    else:
        return False