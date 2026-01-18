import tree
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.utils.naming import auto_name
def standardize_reduction(reduction):
    allowed = {'sum_over_batch_size', 'sum', None, 'none'}
    if reduction not in allowed:
        raise ValueError(f'Invalid value for argument `reduction`. Expected one of {allowed}. Received: reduction={reduction}')
    return reduction