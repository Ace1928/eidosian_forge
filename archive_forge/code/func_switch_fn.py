import tree
from keras.src.backend import KerasTensor
def switch_fn(x):
    if isinstance(x, KerasTensor):
        val = tensor_dict.get(id(x), None)
        if val is not None:
            return val
    return x