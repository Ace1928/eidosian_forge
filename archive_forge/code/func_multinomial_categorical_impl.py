import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_random_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def multinomial_categorical_impl(logits, num_samples, dtype, seed):
    """Implementation for random.categorical (v1) and random.categorical (v2)."""
    logits = ops.convert_to_tensor(logits, name='logits')
    dtype = dtypes.as_dtype(dtype) if dtype else dtypes.int64
    accepted_dtypes = (dtypes.int32, dtypes.int64)
    if dtype not in accepted_dtypes:
        raise ValueError(f'Argument `dtype` got invalid value {dtype}. Accepted dtypes are {accepted_dtypes}.')
    seed1, seed2 = random_seed.get_seed(seed)
    return gen_random_ops.multinomial(logits, num_samples, seed=seed1, seed2=seed2, output_dtype=dtype)