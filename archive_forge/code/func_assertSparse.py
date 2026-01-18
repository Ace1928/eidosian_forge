import json
import shutil
import tempfile
import unittest
import numpy as np
from keras.src import backend
from keras.src import distribution
from keras.src import ops
from keras.src import utils
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.global_state import clear_session
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.models import Model
from keras.src.utils import traceback_utils
from keras.src.utils import tree
def assertSparse(self, x, sparse=True):
    if isinstance(x, KerasTensor):
        self.assertEqual(x.sparse, sparse)
    elif backend.backend() == 'tensorflow':
        import tensorflow as tf
        if sparse:
            self.assertIsInstance(x, tf.SparseTensor)
        else:
            self.assertNotIsInstance(x, tf.SparseTensor)
    elif backend.backend() == 'jax':
        import jax.experimental.sparse as jax_sparse
        if sparse:
            self.assertIsInstance(x, jax_sparse.JAXSparse)
        else:
            self.assertNotIsInstance(x, jax_sparse.JAXSparse)
    else:
        self.assertFalse(sparse, f'Backend {backend.backend()} does not support sparse tensors')