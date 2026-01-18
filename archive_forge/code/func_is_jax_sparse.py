import itertools
import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def is_jax_sparse(x):
    return x.__class__.__module__.startswith('jax.experimental.sparse')