import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import script_ops
def get_iterator_id_fn(unused_dummy):
    """Creates a unique `iterator_id` for each pass over the dataset.

    The returned `iterator_id` disambiguates between multiple concurrently
    existing iterators.

    Args:
      unused_dummy: Ignored value.

    Returns:
      A `tf.int64` tensor whose value uniquely identifies an iterator in
      `generator_state`.
    """
    return script_ops.numpy_function(generator_state.get_next_id, args, dtypes.int64)