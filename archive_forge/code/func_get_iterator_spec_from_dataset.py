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
def get_iterator_spec_from_dataset(strategy, dataset):
    """Returns an iterator spec from dataset function.

  This function constructs type spec for iterator obtained from
  iter(dataset).

  Args:
    strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
    dataset: A tf.data.Dataset instance. If using a function that returns a
      tf.data.Dataset instance, pass dataset_fn.structured_outputs.

  Returns:
    A type_spec for iterator for dataset instance.

  """
    output_element_spec = dataset.element_spec
    if isinstance(dataset._type_spec, (DistributedDatasetSpec, DistributedDatasetsFromFunctionSpec)):
        iterator_type_spec = DistributedIteratorSpec(strategy.extended._input_workers_with_options(), output_element_spec, strategy.extended._container_strategy(), options=None, cardinality=dataset.cardinality, enable_get_next_as_optional=True)
    else:
        if strategy.extended._num_gpus_per_worker:
            logging.warning(f'{strategy.extended._num_gpus_per_worker} GPUs are allocated per worker. Please use DistributedDataset by calling strategy.experimental_distribute_dataset or strategy.distribute_datasets_from_function to make best use of GPU resources')
        iterator_type_spec = iterator_ops.IteratorSpec(output_element_spec)
    return iterator_type_spec