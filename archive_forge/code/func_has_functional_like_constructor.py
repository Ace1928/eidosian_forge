import collections
import copy
import itertools
import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.dtensor import layout_map as layout_map_lib
from keras.src.engine import base_layer
from keras.src.engine import base_layer_utils
from keras.src.engine import functional_utils
from keras.src.engine import input_layer as input_layer_module
from keras.src.engine import input_spec
from keras.src.engine import node as node_module
from keras.src.engine import training as training_lib
from keras.src.engine import training_utils
from keras.src.saving import serialization_lib
from keras.src.saving.legacy import serialization
from keras.src.saving.legacy.saved_model import json_utils
from keras.src.saving.legacy.saved_model import network_serialization
from keras.src.saving.legacy.saved_model import utils as saved_model_utils
from keras.src.utils import generic_utils
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.tools.docs import doc_controls
def has_functional_like_constructor(cls):
    init_args = tf_inspect.getfullargspec(cls.__init__).args[1:]
    functional_init_args = tf_inspect.getfullargspec(Functional.__init__).args[1:]
    if init_args == functional_init_args:
        return True
    return False