import copy
import functools
import re
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
@expects_mask_arg.setter
def expects_mask_arg(self, value):
    self._expects_mask_arg = value