import collections
import contextlib
import functools
import itertools
import threading
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import tf_decorator
def get_model_from_layers(model_layers, input_shape=None, input_dtype=None, name=None, input_ragged=None, input_sparse=None, model_type=None):
    """Builds a model from a sequence of layers.

  Args:
    model_layers: The layers used to build the network.
    input_shape: Shape tuple of the input or 'TensorShape' instance.
    input_dtype: Datatype of the input.
    name: Name for the model.
    input_ragged: Boolean, whether the input data is a ragged tensor.
    input_sparse: Boolean, whether the input data is a sparse tensor.
    model_type: One of "subclass", "subclass_custom_build", "sequential", or
      "functional". When None, defaults to `get_model_type`.

  Returns:
    A Keras model.
  """
    if model_type is None:
        model_type = get_model_type()
    if model_type == 'subclass':
        inputs = None
        if input_ragged or input_sparse:
            inputs = layers.Input(shape=input_shape, dtype=input_dtype, ragged=input_ragged, sparse=input_sparse)
        return _SubclassModel(model_layers, name=name, input_tensor=inputs)
    if model_type == 'subclass_custom_build':
        layer_generating_func = lambda: model_layers
        return _SubclassModelCustomBuild(layer_generating_func, name=name)
    if model_type == 'sequential':
        model = models.Sequential(name=name)
        if input_shape:
            model.add(layers.InputLayer(input_shape=input_shape, dtype=input_dtype, ragged=input_ragged, sparse=input_sparse))
        for layer in model_layers:
            model.add(layer)
        return model
    if model_type == 'functional':
        if not input_shape:
            raise ValueError('Cannot create a functional model from layers with no input shape.')
        inputs = layers.Input(shape=input_shape, dtype=input_dtype, ragged=input_ragged, sparse=input_sparse)
        outputs = inputs
        for layer in model_layers:
            outputs = layer(outputs)
        return models.Model(inputs, outputs, name=name)
    raise ValueError('Unknown model type {}'.format(model_type))