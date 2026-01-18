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
def finalize_fn(iterator_id_t):
    """Releases host-side state for the iterator with ID `iterator_id_t`."""

    def finalize_py_func(iterator_id):
        generator_state.iterator_completed(iterator_id)
        return np.array(0, dtype=np.int64)
    return script_ops.numpy_function(finalize_py_func, [iterator_id_t], dtypes.int64)