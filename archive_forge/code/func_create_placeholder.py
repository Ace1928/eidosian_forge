import os
import warnings
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import model_config
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving import utils_v1 as model_utils
from tensorflow.python.keras.utils import mode_keys
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import save as save_lib
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.util import compat
from tensorflow.python.util import nest
def create_placeholder(spec):
    return backend.placeholder(shape=spec.shape, dtype=spec.dtype, name=spec.name)