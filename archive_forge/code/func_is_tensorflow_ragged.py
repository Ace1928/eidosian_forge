import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.utils import tree
def is_tensorflow_ragged(value):
    if hasattr(value, '__class__'):
        return value.__class__.__name__ == 'RaggedTensor' and 'tensorflow.python.' in str(value.__class__.__module__)
    return False