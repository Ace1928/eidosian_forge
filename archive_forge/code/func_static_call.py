from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
@staticmethod
def static_call(x):
    return x * backend.nn.tanh(backend.nn.softplus(x))