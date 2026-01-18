import numpy as np
from keras.src.api_export import keras_export
from keras.src.backend import config
from keras.src.backend.common import global_state
from keras.src.backend.common.name_scope import current_path
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name
@keras_export('keras.backend.standardize_dtype')
def standardize_dtype(dtype):
    if dtype is None:
        return config.floatx()
    dtype = PYTHON_DTYPES_MAP.get(dtype, dtype)
    if hasattr(dtype, 'name'):
        dtype = dtype.name
    elif hasattr(dtype, '__str__') and ('torch' in str(dtype) or 'jax.numpy' in str(dtype)):
        dtype = str(dtype).split('.')[-1]
    elif hasattr(dtype, '__name__'):
        dtype = dtype.__name__
    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f'Invalid dtype: {dtype}')
    return dtype