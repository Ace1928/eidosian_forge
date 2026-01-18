import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.utils import tree
def is_tensorflow_tensor(value):
    if hasattr(value, '__class__'):
        if value.__class__.__name__ in ('RaggedTensor', 'SparseTensor'):
            return 'tensorflow.python.' in str(value.__class__.__module__)
        for parent in value.__class__.__mro__:
            if parent.__name__ in 'Tensor' and 'tensorflow.python.' in str(parent.__module__):
                return True
    return False