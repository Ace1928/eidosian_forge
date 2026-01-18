import functools
import weakref
import numpy as np
from tensorflow.python.util import nest
def convert_dense_weights_data_format(dense, previous_feature_map_shape, target_data_format='channels_first'):
    """Utility useful when changing a convnet's `data_format`.

  When porting the weights of a convnet from one data format to the other,
  if the convnet includes a `Flatten` layer
  (applied to the last convolutional feature map)
  followed by a `Dense` layer, the weights of that `Dense` layer
  should be updated to reflect the new dimension ordering.

  Args:
      dense: The target `Dense` layer.
      previous_feature_map_shape: A shape tuple of 3 integers,
          e.g. `(512, 7, 7)`. The shape of the convolutional
          feature map right before the `Flatten` layer that
          came before the target `Dense` layer.
      target_data_format: One of "channels_last", "channels_first".
          Set it "channels_last"
          if converting a "channels_first" model to "channels_last",
          or reciprocally.
  """
    assert target_data_format in {'channels_last', 'channels_first'}
    kernel, bias = dense.get_weights()
    for i in range(kernel.shape[1]):
        if target_data_format == 'channels_first':
            c, h, w = previous_feature_map_shape
            original_fm_shape = (h, w, c)
            ki = kernel[:, i].reshape(original_fm_shape)
            ki = np.transpose(ki, (2, 0, 1))
        else:
            h, w, c = previous_feature_map_shape
            original_fm_shape = (c, h, w)
            ki = kernel[:, i].reshape(original_fm_shape)
            ki = np.transpose(ki, (1, 2, 0))
        kernel[:, i] = np.reshape(ki, (np.prod(previous_feature_map_shape),))
    dense.set_weights([kernel, bias])