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
def read_up_to(self, queue, num_records, name=None):
    """Returns up to num_records (key, value) pairs produced by a reader.

    Will dequeue a work unit from queue if necessary (e.g., when the
    Reader needs to start reading from a new file since it has
    finished with the previous file).
    It may return less than num_records even before the last batch.

    Args:
      queue: A Queue or a mutable string Tensor representing a handle
        to a Queue, with string work items.
      num_records: Number of records to read.
      name: A name for the operation (optional).

    Returns:
      A tuple of Tensors (keys, values).
      keys: A 1-D string Tensor.
      values: A 1-D string Tensor.
    """
    if isinstance(queue, tensor_lib.Tensor):
        queue_ref = queue
    else:
        queue_ref = queue.queue_ref
    if self._reader_ref.dtype == dtypes.resource:
        return gen_io_ops.reader_read_up_to_v2(self._reader_ref, queue_ref, num_records, name=name)
    else:
        old_queue_op = gen_data_flow_ops.fake_queue(queue_ref)
        return gen_io_ops.reader_read_up_to(self._reader_ref, old_queue_op, num_records, name=name)