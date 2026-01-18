import itertools
import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def scipy_sparse_to_jax_sparse(x):
    import jax.experimental.sparse as jax_sparse
    coo = x.tocoo()
    indices = np.concatenate((np.expand_dims(coo.row, 1), np.expand_dims(coo.col, 1)), axis=1)
    return jax_sparse.BCOO((coo.data, indices), shape=coo.shape)