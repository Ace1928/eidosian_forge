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
@tf_export('signal.mdct')
@dispatch.add_dispatch_support
def mdct(signals, frame_length, window_fn=window_ops.vorbis_window, pad_end=False, norm=None, name=None):
    """Computes the [Modified Discrete Cosine Transform][mdct] of `signals`.

  Implemented with TPU/GPU-compatible ops and supports gradients.

  Args:
    signals: A `[..., samples]` `float32`/`float64` `Tensor` of real-valued
      signals.
    frame_length: An integer scalar `Tensor`. The window length in samples
      which must be divisible by 4.
    window_fn: A callable that takes a frame_length and a `dtype` keyword
      argument and returns a `[frame_length]` `Tensor` of samples in the
      provided datatype. If set to `None`, a rectangular window with a scale of
      1/sqrt(2) is used. For perfect reconstruction of a signal from `mdct`
      followed by `inverse_mdct`, please use `tf.signal.vorbis_window`,
      `tf.signal.kaiser_bessel_derived_window` or `None`. If using another
      window function, make sure that w[n]^2 + w[n + frame_length // 2]^2 = 1
      and w[n] = w[frame_length - n - 1] for n = 0,...,frame_length // 2 - 1 to
      achieve perfect reconstruction.
    pad_end: Whether to pad the end of `signals` with zeros when the provided
      frame length and step produces a frame that lies partially past its end.
    norm: If it is None, unnormalized dct4 is used, if it is "ortho"
      orthonormal dct4 is used.
    name: An optional name for the operation.

  Returns:
    A `[..., frames, frame_length // 2]` `Tensor` of `float32`/`float64`
    MDCT values where `frames` is roughly `samples // (frame_length // 2)`
    when `pad_end=False`.

  Raises:
    ValueError: If `signals` is not at least rank 1, `frame_length` is
      not scalar, or `frame_length` is not a multiple of `4`.

  [mdct]: https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform
  """
    with ops.name_scope(name, 'mdct', [signals, frame_length]):
        signals = ops.convert_to_tensor(signals, name='signals')
        signals.shape.with_rank_at_least(1)
        frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
        frame_length.shape.assert_has_rank(0)
        frame_length_static = tensor_util.constant_value(frame_length)
        if frame_length_static is not None:
            if frame_length_static % 4 != 0:
                raise ValueError('The frame length must be a multiple of 4.')
            frame_step = ops.convert_to_tensor(frame_length_static // 2, dtype=frame_length.dtype)
        else:
            frame_step = frame_length // 2
        framed_signals = shape_ops.frame(signals, frame_length, frame_step, pad_end=pad_end)
        if window_fn is not None:
            window = window_fn(frame_length, dtype=framed_signals.dtype)
            framed_signals *= window
        else:
            framed_signals *= 1.0 / np.sqrt(2)
        split_frames = array_ops.split(framed_signals, 4, axis=-1)
        frame_firsthalf = -array_ops.reverse(split_frames[2], [-1]) - split_frames[3]
        frame_secondhalf = split_frames[0] - array_ops.reverse(split_frames[1], [-1])
        frames_rearranged = array_ops.concat((frame_firsthalf, frame_secondhalf), axis=-1)
        return dct_ops.dct(frames_rearranged, type=4, norm=norm)