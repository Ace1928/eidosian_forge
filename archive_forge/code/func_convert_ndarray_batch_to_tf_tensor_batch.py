from typing import TYPE_CHECKING, Dict, List, Optional, Union, Tuple
import numpy as np
import pyarrow
import tensorflow as tf
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
def convert_ndarray_batch_to_tf_tensor_batch(ndarrays: Union[np.ndarray, Dict[str, np.ndarray]], dtypes: Optional[Union[tf.dtypes.DType, Dict[str, tf.dtypes.DType]]]=None) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
    """Convert a NumPy ndarray batch to a TensorFlow Tensor batch.

    Args:
        ndarray: A (dict of) NumPy ndarray(s) that we wish to convert to a TensorFlow
            Tensor.
        dtype: A (dict of) TensorFlow dtype(s) for the created tensor; if None, the
            dtype will be inferred from the NumPy ndarray data.

    Returns: A (dict of) TensorFlow Tensor(s).
    """
    if isinstance(ndarrays, np.ndarray):
        if isinstance(dtypes, dict):
            if len(dtypes) != 1:
                raise ValueError(f'When constructing a single-tensor batch, only a single dtype should be given, instead got: {dtypes}')
            dtypes = next(iter(dtypes.values()))
        batch = convert_ndarray_to_tf_tensor(ndarrays, dtypes)
    else:
        batch = {col_name: convert_ndarray_to_tf_tensor(col_ndarray, dtype=dtypes[col_name] if isinstance(dtypes, dict) else dtypes) for col_name, col_ndarray in ndarrays.items()}
    return batch