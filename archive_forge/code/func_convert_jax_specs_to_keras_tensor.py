import types
import jax
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import tree
from jax.tree_util import Partial
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.jax import distribution_lib
def convert_jax_specs_to_keras_tensor(x1, x2):
    if isinstance(x1, jax.ShapeDtypeStruct):
        if not isinstance(x2, jax.ShapeDtypeStruct):
            raise ValueError('Indeterministic output ordering.')
        return KerasTensor(merge_shapes(x1.shape, x2.shape), dtype=x1.dtype)
    elif isinstance(x1, jax_sparse.BCOO):
        if not isinstance(x2, jax_sparse.BCOO):
            raise ValueError('Indeterministic output ordering.')
        return KerasTensor(merge_shapes(x1.shape, x2.shape), dtype=x1.dtype, sparse=True)
    else:
        return x1