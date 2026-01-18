import tensorflow.compat.v2 as tf
from keras.src.utils import control_flow_util
from tensorflow.python.platform import tf_logging as logging
def generate_zero_filled_state(batch_size_tensor, state_size, dtype):
    """Generate a zero filled tensor with shape [batch_size, state_size]."""
    if batch_size_tensor is None or dtype is None:
        raise ValueError(f'batch_size and dtype cannot be None while constructing initial state. Received: batch_size={batch_size_tensor}, dtype={dtype}')

    def create_zeros(unnested_state_size):
        flat_dims = tf.TensorShape(unnested_state_size).as_list()
        init_state_size = [batch_size_tensor] + flat_dims
        return tf.zeros(init_state_size, dtype=dtype)
    if tf.nest.is_nested(state_size):
        return tf.nest.map_structure(create_zeros, state_size)
    else:
        return create_zeros(state_size)