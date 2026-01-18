import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import dct_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.ops.signal import reconstruction_ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.ops.signal import window_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def inverse_stft_window_fn_inner(frame_length, dtype):
    """Computes a window that can be used in `inverse_stft`.

    Args:
      frame_length: An integer scalar `Tensor`. The window length in samples.
      dtype: Data type of waveform passed to `stft`.

    Returns:
      A window suitable for reconstructing original waveform in `inverse_stft`.

    Raises:
      ValueError: If `frame_length` is not scalar, `forward_window_fn` is not a
      callable that takes a window length and a `dtype` keyword argument and
      returns a `[window_length]` `Tensor` of samples in the provided datatype
      `frame_step` is not scalar, or `frame_step` is not scalar.
    """
    with ops.name_scope(name, 'inverse_stft_window_fn', [forward_window_fn]):
        frame_step_ = ops.convert_to_tensor(frame_step, name='frame_step')
        frame_step_.shape.assert_has_rank(0)
        frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
        frame_length.shape.assert_has_rank(0)
        forward_window = forward_window_fn(frame_length, dtype=dtype)
        denom = math_ops.square(forward_window)
        overlaps = -(-frame_length // frame_step_)
        denom = array_ops.pad(denom, [(0, overlaps * frame_step_ - frame_length)])
        denom = array_ops.reshape(denom, [overlaps, frame_step_])
        denom = math_ops.reduce_sum(denom, 0, keepdims=True)
        denom = array_ops.tile(denom, [overlaps, 1])
        denom = array_ops.reshape(denom, [overlaps * frame_step_])
        return forward_window / denom[:frame_length]