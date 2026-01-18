import sys
from typing import Callable
from typing import Dict
from typing import List
from typing import Union
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def allow_bigger_recursion(target_limit: int):
    """Increase default recursion limit to create larger models."""
    current_limit = sys.getrecursionlimit()
    if current_limit < target_limit:
        sys.setrecursionlimit(target_limit)