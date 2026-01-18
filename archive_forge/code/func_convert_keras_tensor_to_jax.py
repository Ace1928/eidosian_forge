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
def convert_keras_tensor_to_jax(x, fill_value=None):
    if isinstance(x, KerasTensor):
        shape = list(x.shape)
        if fill_value:
            for i, e in enumerate(shape):
                if e is None:
                    shape[i] = fill_value
        jax_tensor = jax.ShapeDtypeStruct(shape, dtype=x.dtype)
        return jax_tensor
    if isinstance(x, types.FunctionType):

        def _fn(*args, **kwargs):
            out = x(*args, **kwargs)
            out = convert_keras_tensor_to_jax(out, fill_value=fill_value)
            return out
        return Partial(_fn)
    if isinstance(x, dict):
        return {k: convert_keras_tensor_to_jax(v, fill_value=fill_value) for k, v in x.items()}
    if isinstance(x, list):
        return [convert_keras_tensor_to_jax(xi, fill_value=fill_value) for xi in x]
    return x