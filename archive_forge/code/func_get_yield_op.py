import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def get_yield_op(self):
    """Adds a node that yields a group of records every time it is executed.
    If RecordInput `batches` parameter is not None, it yields a list of
    record batches with the specified `batch_size`.
    """
    compression_type = python_io.TFRecordOptions.get_compression_type_string(python_io.TFRecordOptions(self._compression_type))
    records = gen_data_flow_ops.record_input(file_pattern=self._file_pattern, file_buffer_size=self._buffer_size, file_parallelism=self._parallelism, file_shuffle_shift_ratio=self._shift_ratio, batch_size=self._batch_size, file_random_seed=self._seed, compression_type=compression_type, name=self._name)
    if self._batches is None:
        return records
    else:
        with ops.name_scope(self._name):
            batch_list = [[] for _ in range(self._batches)]
            records = array_ops.split(records, self._batch_size, 0)
            for index, protobuf in enumerate(records):
                batch_index = index % self._batches
                batch_list[batch_index].append(array_ops.reshape(protobuf, []))
            return batch_list