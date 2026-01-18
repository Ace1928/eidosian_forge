import functools
import sys
import traceback
import numpy as np
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.operators import variables
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import misc
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.types import distribute
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
def guarded_extra_test():
    extra_test_result = extra_test()
    try:
        return bool(extra_test_result)
    except errors_impl.OperatorNotAllowedInGraphError as e:
        ag_logging.log(1, 'Caught error while evaluating loop stop condition', exc_info=True)
        raise NotImplementedError('break and return statements which depend on a TF condition are not supported in Python for loops. Did you intend to make it a TF loop?\nSee https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#consistency-of-control-flow-types for more info.') from e