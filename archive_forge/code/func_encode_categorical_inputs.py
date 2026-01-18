import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import tf_utils
def encode_categorical_inputs(inputs, output_mode, depth, dtype='float32', sparse=False, count_weights=None, idf_weights=None):
    """Encodes categoical inputs according to output_mode."""
    if output_mode == INT:
        return tf.identity(tf.cast(inputs, dtype))
    original_shape = inputs.shape
    if inputs.shape.rank == 0:
        inputs = expand_dims(inputs, -1)
    if output_mode == ONE_HOT:
        if inputs.shape[-1] != 1:
            inputs = expand_dims(inputs, -1)
    if inputs.shape.rank > 2:
        raise ValueError(f"When output_mode is not `'int'`, maximum supported output rank is 2. Received output_mode {output_mode} and input shape {original_shape}, which would result in output rank {inputs.shape.rank}.")
    binary_output = output_mode in (MULTI_HOT, ONE_HOT)
    if sparse:
        bincounts = sparse_bincount(inputs, depth, binary_output, dtype, count_weights)
    else:
        bincounts = dense_bincount(inputs, depth, binary_output, dtype, count_weights)
    if output_mode != TF_IDF:
        return bincounts
    if idf_weights is None:
        raise ValueError(f"When output mode is `'tf_idf'`, idf_weights must be provided. Received: output_mode={output_mode} and idf_weights={idf_weights}")
    if sparse:
        value_weights = tf.gather(idf_weights, bincounts.indices[:, -1])
        return tf.SparseTensor(bincounts.indices, value_weights * bincounts.values, bincounts.dense_shape)
    else:
        return tf.multiply(bincounts, idf_weights)