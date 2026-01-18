import io
from packaging.version import parse
from keras.src.api_export import keras_export
from keras.src.layers import Layer
from keras.src.ops import convert_to_numpy
from keras.src.ops import convert_to_tensor
def no_grad(orig_func):
    import torch
    if parse(torch.__version__) >= parse('2.1.0'):
        return torch.no_grad(orig_func)
    else:
        return orig_func