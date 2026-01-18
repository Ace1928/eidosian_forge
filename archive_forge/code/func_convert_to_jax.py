import itertools
import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def convert_to_jax(x):
    if is_scipy_sparse(x):
        return scipy_sparse_to_jax_sparse(x)
    elif is_tf_sparse(x):
        return tf_sparse_to_jax_sparse(x)
    return convert_to_tensor(x)