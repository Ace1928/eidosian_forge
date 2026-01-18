import functools
import math
import numpy as np
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
from keras.src.utils import tree
def grab_one(x):
    if isinstance(x, array_slicing.TensorflowSparseWrapper):
        return array_slicing.slice_tensorflow_sparse_wrapper(x, i)
    if isinstance(x, (list, tuple, dict)):
        return None
    if tf.is_tensor(x):
        return tf.gather(x, i, axis=0)
    return x