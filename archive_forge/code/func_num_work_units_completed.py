from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops.gen_io_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def num_work_units_completed(self, name=None):
    """Returns the number of work units this reader has finished processing.

    Args:
      name: A name for the operation (optional).

    Returns:
      An int64 Tensor.
    """
    if self._reader_ref.dtype == dtypes.resource:
        return gen_io_ops.reader_num_work_units_completed_v2(self._reader_ref, name=name)
    else:
        return gen_io_ops.reader_num_work_units_completed(self._reader_ref, name=name)