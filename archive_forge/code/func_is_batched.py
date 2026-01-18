import multiprocessing
import os
import random
import time
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def is_batched(tf_dataset):
    """ "Check if the `tf.data.Dataset` is batched."""
    return hasattr(tf_dataset, '_batch_size')