import math
import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
from keras.src.utils.dataset_utils import is_torch_tensor
from keras.src.utils.nest import lists_to_tuples
def convert_to_arrays(arrays):
    """Process array-like inputs.

    This function:

    - Converts tf.Tensors to NumPy arrays.
    - Converts `pandas.Series` to `np.ndarray`
    - Converts `list`s to `tuple`s (for `tf.data` support).

    Args:
        inputs: Structure of `Tensor`s, NumPy arrays, or tensor-like.

    Returns:
        Structure of NumPy `ndarray`s.
    """

    def convert_single_array(x):
        if x is None:
            return x
        if pandas is not None:
            if isinstance(x, pandas.Series):
                x = np.expand_dims(x.to_numpy(), axis=-1)
            elif isinstance(x, pandas.DataFrame):
                x = x.to_numpy()
        if is_tf_ragged_tensor(x):
            from keras.src.utils.module_utils import tensorflow as tf
            if backend.is_float_dtype(x.dtype) and (not backend.standardize_dtype(x.dtype) == backend.floatx()):
                x = tf.cast(x, backend.floatx())
            return x
        if not isinstance(x, np.ndarray):
            if hasattr(x, '__array__'):
                if is_torch_tensor(x):
                    x = x.cpu()
                x = np.asarray(x)
            else:
                raise ValueError(f'Expected a NumPy array, tf.Tensor, tf.RaggedTensor, jax.np.ndarray, torch.Tensor, Pandas Dataframe, or Pandas Series. Received invalid input: {x} (of type {type(x)})')
        if x.dtype == object:
            return x
        if backend.is_float_dtype(x.dtype) and (not backend.standardize_dtype(x.dtype) == backend.floatx()):
            x = x.astype(backend.floatx())
        return x
    arrays = tree.map_structure(convert_single_array, arrays)
    return lists_to_tuples(arrays)