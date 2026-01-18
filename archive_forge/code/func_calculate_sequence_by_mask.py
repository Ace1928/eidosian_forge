import uuid
import tensorflow.compat.v2 as tf
from tensorflow.python.eager.context import get_device_name
def calculate_sequence_by_mask(mask, time_major):
    """Calculate the sequence length tensor (1-D) based on the masking tensor.

    The masking tensor is a 2D boolean tensor with shape [batch, timestep]. For
    any timestep that should be masked, the corresponding field will be False.
    Consider the following example:
      a = [[True, True, False, False],
           [True, True, True, False]]
    It is a (2, 4) tensor, and the corresponding sequence length result should
    be 1D tensor with value [2, 3]. Note that the masking tensor must be right
    padded that could be checked by, e.g., `is_sequence_right_padded()`.

    Args:
      mask: Boolean tensor with shape [batch, timestep] or [timestep, batch] if
        time_major=True.
      time_major: Boolean, which indicates whether the mask is time major or
        batch major.
    Returns:
      sequence_length: 1D int32 tensor.
    """
    timestep_index = 0 if time_major else 1
    return tf.reduce_sum(tf.cast(mask, tf.int32), axis=timestep_index)