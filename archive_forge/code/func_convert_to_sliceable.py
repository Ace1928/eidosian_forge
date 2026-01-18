import collections
import math
import numpy as np
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import tree
def convert_to_sliceable(arrays, target_backend=None):
    """Convert a structure of arrays into `Sliceable` instances

    Args:
        arrays: the arrays to convert.
        target_backend: the target backend for the output:
            - `None` indicates that `arrays` will be wrapped into `Sliceable`s
              as-is without using a different representation. This is used by
              `train_validation_split()`.
            - `tensorflow` indicates that
              `Sliceable.convert_to_tf_dataset_compatible` will be called. The
              returned structure therefore contains arrays, not `Sliceable`s.
            - `numpy`, `jax` or `torch` indices that the arrays will eventually
              be converted to this backend type after slicing. In this case,
              the intermediary `Sliceable`s may use a different representation
              from the input `arrays` for better performance.
    Returns: the same structure with `Sliceable` instances or arrays.
    """

    def convert_single_array(x):
        if x is None:
            return x
        if isinstance(x, np.ndarray):
            sliceable_class = NumpySliceable
        elif data_adapter_utils.is_tensorflow_tensor(x):
            if data_adapter_utils.is_tensorflow_ragged(x):
                sliceable_class = TensorflowRaggedSliceable
            elif data_adapter_utils.is_tensorflow_sparse(x):
                sliceable_class = TensorflowSparseSliceable
            else:
                sliceable_class = TensorflowSliceable
        elif data_adapter_utils.is_jax_array(x):
            if data_adapter_utils.is_jax_sparse(x):
                sliceable_class = JaxSparseSliceable
            else:
                sliceable_class = JaxSliceable
        elif data_adapter_utils.is_torch_tensor(x):
            sliceable_class = TorchSliceable
        elif pandas is not None and isinstance(x, pandas.DataFrame):
            sliceable_class = PandasDataFrameSliceable
        elif pandas is not None and isinstance(x, pandas.Series):
            sliceable_class = PandasSeriesSliceable
        elif data_adapter_utils.is_scipy_sparse(x):
            sliceable_class = ScipySparseSliceable
        elif hasattr(x, '__array__'):
            x = np.asarray(x)
            sliceable_class = NumpySliceable
        else:
            raise ValueError(f'Expected a NumPy array, tf.Tensor, tf.RaggedTensor, tf.SparseTensor, jax.np.ndarray, jax.experimental.sparse.JAXSparse, torch.Tensor, Pandas Dataframe, or Pandas Series. Received invalid input: {x} (of type {type(x)})')

        def is_non_floatx_float(dtype):
            return not dtype == object and backend.is_float_dtype(dtype) and (not backend.standardize_dtype(dtype) == backend.floatx())
        cast_dtype = None
        if pandas is not None and isinstance(x, pandas.DataFrame):
            if any((is_non_floatx_float(d) for d in x.dtypes.values)):
                cast_dtype = backend.floatx()
        elif is_non_floatx_float(x.dtype):
            cast_dtype = backend.floatx()
        if cast_dtype is not None:
            x = sliceable_class.cast(x, cast_dtype)
        if target_backend is None:
            return sliceable_class(x)
        if target_backend == 'tensorflow':
            return sliceable_class.convert_to_tf_dataset_compatible(x)
        if sliceable_class == JaxSliceable or (target_backend == 'jax' and sliceable_class in (TensorflowSliceable, TorchSliceable)):
            x = np.asarray(x)
            sliceable_class = NumpySliceable
        return sliceable_class(x)
    return tree.map_structure(convert_single_array, arrays)