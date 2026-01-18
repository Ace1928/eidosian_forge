import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.utils import conv_utils
def local_conv_matmul(inputs, kernel, kernel_mask, output_shape):
    """Apply N-D convolution with un-shared weights using a single matmul call.

    This method outputs `inputs . (kernel * kernel_mask)`
    (with `.` standing for matrix-multiply and `*` for element-wise multiply)
    and requires a precomputed `kernel_mask` to zero-out weights in `kernel` and
    hence perform the same operation as a convolution with un-shared
    (the remaining entries in `kernel`) weights. It also does the necessary
    reshapes to make `inputs` and `kernel` 2-D and `output` (N+2)-D.

    Args:
        inputs: (N+2)-D tensor with shape `(batch_size, channels_in, d_in1, ...,
          d_inN)` or `(batch_size, d_in1, ..., d_inN, channels_in)`.
        kernel: the unshared weights for N-D convolution,
            an (N+2)-D tensor of shape: `(d_in1, ..., d_inN, channels_in,
            d_out2, ..., d_outN, channels_out)` or `(channels_in, d_in1, ...,
            d_inN, channels_out, d_out2, ..., d_outN)`, with the ordering of
            channels and spatial dimensions matching that of the input. Each
            entry is the weight between a particular input and output location,
            similarly to a fully-connected weight matrix.
        kernel_mask: a float 0/1 mask tensor of shape: `(d_in1, ..., d_inN, 1,
          d_out2, ..., d_outN, 1)` or `(1, d_in1, ..., d_inN, 1, d_out2, ...,
          d_outN)`, with the ordering of singleton and spatial dimensions
          matching that of the input. Mask represents the connectivity pattern
          of the layer and is precomputed elsewhere based on layer parameters:
          stride, padding, and the receptive field shape.
        output_shape: a tuple of (N+2) elements representing the output shape:
          `(batch_size, channels_out, d_out1, ..., d_outN)` or `(batch_size,
          d_out1, ..., d_outN, channels_out)`, with the ordering of channels and
          spatial dimensions matching that of the input.

    Returns:
        Output (N+2)-D tensor with shape `output_shape`.
    """
    inputs_flat = backend.reshape(inputs, (backend.shape(inputs)[0], -1))
    kernel = kernel_mask * kernel
    kernel = make_2d(kernel, split_dim=backend.ndim(kernel) // 2)
    output_flat = tf.matmul(inputs_flat, kernel, b_is_sparse=True)
    output = backend.reshape(output_flat, [backend.shape(output_flat)[0]] + output_shape.as_list()[1:])
    return output