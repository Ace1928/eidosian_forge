import tree
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.utils.naming import auto_name
def reduce_values(values, reduction='sum_over_batch_size'):
    if reduction is None or reduction == 'none' or tuple(values.shape) == () or (tuple(values.shape) == (0,)):
        return values
    loss = ops.sum(values)
    if reduction == 'sum_over_batch_size':
        loss /= ops.cast(ops.prod(ops.convert_to_tensor(ops.shape(values), dtype='int32')), loss.dtype)
    return loss