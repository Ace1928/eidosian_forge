import itertools
import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def jax_sparse_to_tf_sparse(x):
    from keras.src.utils.module_utils import tensorflow as tf
    return tf.SparseTensor(x.indices, x.data, x.shape)