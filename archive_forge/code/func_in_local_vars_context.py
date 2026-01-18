import collections
import contextlib
import copy
import platform
import random
import threading
import numpy as np
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import backend
from keras.src.engine import keras_tensor
from keras.src.utils import object_identity
from keras.src.utils import tf_contextlib
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python import pywrap_tfe
def in_local_vars_context():
    ctx = get_metric_local_vars_scope()
    return ctx is not None