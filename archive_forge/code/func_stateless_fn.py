import inspect
import itertools
import string
from absl import logging
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.layers import Layer
from keras.src.models import Functional
from keras.src.models import Sequential
from keras.src.utils import io_utils
from keras.src.utils import tree
from keras.src.utils.module_utils import tensorflow as tf
def stateless_fn(variables, *args, **kwargs):
    state_mapping = zip(self._backend_variables, variables)
    with StatelessScope(state_mapping=state_mapping):
        return fn(*args, **kwargs)