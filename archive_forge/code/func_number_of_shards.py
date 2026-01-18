import itertools
import numpy as np
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_sharding
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest
@property
def number_of_shards(self):
    """Gets the number of shards to use for the InfeedQueue.

    Returns:
      Number of shards or None if the number of shards has not been set.
    """
    return self._sharding_policies[0].number_of_shards