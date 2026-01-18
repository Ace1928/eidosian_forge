import collections
import math
import numpy as np
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import tree
@classmethod
def convert_to_jax_compatible(cls, x):
    return data_adapter_utils.scipy_sparse_to_jax_sparse(x)