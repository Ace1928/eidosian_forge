from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
def convert_input(self, x, autocast, dtype):
    dtype = backend.standardize_dtype(dtype)
    if backend.is_tensor(x):
        if autocast and backend.is_float_dtype(x.dtype) and (x.dtype != dtype):
            x = backend.cast(x, dtype=dtype)
        return x
    elif backend.is_keras_tensor(x):
        if autocast and backend.is_float_dtype(x.dtype) and (x.dtype != dtype):
            x.dtype = dtype
        return x
    elif hasattr(x, '__array__'):
        return ops.convert_to_tensor(x, dtype=dtype)
    return x