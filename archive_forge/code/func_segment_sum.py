import functools
import typing
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@dispatch.dispatch_for_api(math_ops.unsorted_segment_sum)
def segment_sum(data: ragged_tensor.RaggedOrDense, segment_ids: ragged_tensor.RaggedOrDense, num_segments, name=None):
    return _ragged_segment_aggregate(math_ops.unsorted_segment_sum, data=data, segment_ids=segment_ids, num_segments=num_segments, name=name or 'RaggedSegmentSum')