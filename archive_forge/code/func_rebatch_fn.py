import functools
import sys
import time
import six
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.distribute_lib import InputReplicationMode
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import distribute as distribute_types
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
def rebatch_fn(dataset, worker_index):
    try:

        def apply_rebatch():
            batch_sizes = distribute.batch_sizes_for_worker(batch_size, num_workers, num_replicas_per_worker, worker_index)
            return dataset.rebatch(batch_sizes).prefetch(num_replicas_per_worker)

        def apply_legacy_rebatch():
            return distribute._LegacyRebatchDataset(dataset, num_replicas_in_sync).prefetch(num_replicas_per_worker)
        with ops.colocate_with(dataset._variant_tensor):
            return tf_cond.cond(math_ops.not_equal(batch_size, -1), true_fn=apply_rebatch, false_fn=apply_legacy_rebatch)
    except errors.InvalidArgumentError as e:
        if 'without encountering a batch' in str(e):
            six.reraise(ValueError, ValueError('Call the `batch` method on the input Dataset in order to be able to split your input across {} replicas.\n Please see the tf.distribute.Strategy guide. {}'.format(num_replicas_in_sync, e)), sys.exc_info()[2])
        else:
            raise