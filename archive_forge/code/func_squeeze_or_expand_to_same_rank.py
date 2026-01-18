import tree
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.utils.naming import auto_name
def squeeze_or_expand_to_same_rank(x1, x2, expand_rank_1=True):
    """Squeeze/expand last dim if ranks differ from expected by exactly 1."""
    x1_rank = len(x1.shape)
    x2_rank = len(x2.shape)
    if x1_rank == x2_rank:
        return (x1, x2)
    if x1_rank == x2_rank + 1:
        if x1.shape[-1] == 1:
            if x2_rank == 1 and expand_rank_1:
                x2 = ops.expand_dims(x2, axis=-1)
            else:
                x1 = ops.squeeze(x1, axis=-1)
    if x2_rank == x1_rank + 1:
        if x2.shape[-1] == 1:
            if x1_rank == 1 and expand_rank_1:
                x1 = ops.expand_dims(x1, axis=-1)
            else:
                x2 = ops.squeeze(x2, axis=-1)
    return (x1, x2)