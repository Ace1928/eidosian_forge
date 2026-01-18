import itertools
import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def scipy_sparse_to_tf_sparse(x):
    from keras.src.utils.module_utils import tensorflow as tf
    coo = x.tocoo()
    indices = np.concatenate((np.expand_dims(coo.row, 1), np.expand_dims(coo.col, 1)), axis=1)
    return tf.SparseTensor(indices, coo.data, coo.shape)