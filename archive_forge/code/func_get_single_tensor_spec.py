import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.utils import tree
def get_single_tensor_spec(*tensors):
    x = tensors[0]
    rank = len(x.shape)
    if rank < 1:
        raise ValueError(f'When passing a dataset to a Keras model, the arrays must be at least rank 1. Received: {x} of rank {len(x.shape)}.')
    for t in tensors:
        if len(t.shape) != rank:
            raise ValueError(f'When passing a dataset to a Keras model, the corresponding arrays in each batch must have the same rank. Received: {x} and {t}')
    shape = []
    for dims in zip(*[list(x.shape) for x in tensors]):
        dims_set = set(dims)
        shape.append(dims_set.pop() if len(dims_set) == 1 else None)
    shape[0] = None
    dtype = backend.standardize_dtype(x.dtype)
    if isinstance(x, tf.RaggedTensor):
        return tf.RaggedTensorSpec(shape=shape, dtype=dtype)
    if isinstance(x, tf.SparseTensor) or is_scipy_sparse(x) or is_jax_sparse(x):
        return tf.SparseTensorSpec(shape=shape, dtype=dtype)
    else:
        return tf.TensorSpec(shape=shape, dtype=dtype)