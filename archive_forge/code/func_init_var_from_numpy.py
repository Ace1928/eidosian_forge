import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.util import nest
def init_var_from_numpy(input_var, numpy_input, session):
    """Initialize `input_var` to `numpy_input` using `session` in graph mode."""
    with ops.init_scope():
        if context.executing_eagerly():
            input_var.assign(numpy_input)
            return
        assert session is not None
        session.run(input_var.initializer)
        start_placeholder = array_ops.placeholder(dtypes.int64, ())
        end_placeholder = array_ops.placeholder(dtypes.int64, ())
        slice_placeholder = array_ops.placeholder(input_var.dtype)
        assign_slice_op = input_var[start_placeholder:end_placeholder].assign(slice_placeholder)
        byte_size_per_batch_element = np.prod(numpy_input.shape[1:]) * input_var.dtype.size
        batch_size_per_slice = int(np.ceil((64 << 20) / byte_size_per_batch_element))
        start = 0
        limit = numpy_input.shape[0]
        while start < limit:
            end = min(start + batch_size_per_slice, limit)
            session.run(assign_slice_op, feed_dict={start_placeholder: start, end_placeholder: end, slice_placeholder: numpy_input[start:end]})
            start = end