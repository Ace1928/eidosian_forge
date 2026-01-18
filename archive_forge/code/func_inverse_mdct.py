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
@tf_export('signal.inverse_mdct')
@dispatch.add_dispatch_support
def inverse_mdct(mdcts, window_fn=window_ops.vorbis_window, norm=None, name=None):
    """Computes the inverse modified DCT of `mdcts`.

  To reconstruct an original waveform, the same window function should
  be used with `mdct` and `inverse_mdct`.

  Example usage:

  >>> @tf.function
  ... def compare_round_trip():
  ...   samples = 1000
  ...   frame_length = 400
  ...   halflen = frame_length // 2
  ...   waveform = tf.random.normal(dtype=tf.float32, shape=[samples])
  ...   waveform_pad = tf.pad(waveform, [[halflen, 0],])
  ...   mdct = tf.signal.mdct(waveform_pad, frame_length, pad_end=True,
  ...                         window_fn=tf.signal.vorbis_window)
  ...   inverse_mdct = tf.signal.inverse_mdct(mdct,
  ...                                         window_fn=tf.signal.vorbis_window)
  ...   inverse_mdct = inverse_mdct[halflen: halflen + samples]
  ...   return waveform, inverse_mdct
  >>> waveform, inverse_mdct = compare_round_trip()
  >>> np.allclose(waveform.numpy(), inverse_mdct.numpy(), rtol=1e-3, atol=1e-4)
  True

  Implemented with TPU/GPU-compatible ops and supports gradients.

  Args:
    mdcts: A `float32`/`float64` `[..., frames, frame_length // 2]`
      `Tensor` of MDCT bins representing a batch of `frame_length // 2`-point
      MDCTs.
    window_fn: A callable that takes a frame_length and a `dtype` keyword
      argument and returns a `[frame_length]` `Tensor` of samples in the
      provided datatype. If set to `None`, a rectangular window with a scale of
      1/sqrt(2) is used. For perfect reconstruction of a signal from `mdct`
      followed by `inverse_mdct`, please use `tf.signal.vorbis_window`,
      `tf.signal.kaiser_bessel_derived_window` or `None`. If using another
      window function, make sure that w[n]^2 + w[n + frame_length // 2]^2 = 1
      and w[n] = w[frame_length - n - 1] for n = 0,...,frame_length // 2 - 1 to
      achieve perfect reconstruction.
    norm: If "ortho", orthonormal inverse DCT4 is performed, if it is None,
      a regular dct4 followed by scaling of `1/frame_length` is performed.
    name: An optional name for the operation.

  Returns:
    A `[..., samples]` `Tensor` of `float32`/`float64` signals representing
    the inverse MDCT for each input MDCT in `mdcts` where `samples` is
    `(frames - 1) * (frame_length // 2) + frame_length`.

  Raises:
    ValueError: If `mdcts` is not at least rank 2.

  [mdct]: https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform
  """
    with ops.name_scope(name, 'inverse_mdct', [mdcts]):
        mdcts = ops.convert_to_tensor(mdcts, name='mdcts')
        mdcts.shape.with_rank_at_least(2)
        half_len = math_ops.cast(mdcts.shape[-1], dtype=dtypes.int32)
        if norm is None:
            half_len_float = math_ops.cast(half_len, dtype=mdcts.dtype)
            result_idct4 = 0.5 / half_len_float * dct_ops.dct(mdcts, type=4)
        elif norm == 'ortho':
            result_idct4 = dct_ops.dct(mdcts, type=4, norm='ortho')
        split_result = array_ops.split(result_idct4, 2, axis=-1)
        real_frames = array_ops.concat((split_result[1], -array_ops.reverse(split_result[1], [-1]), -array_ops.reverse(split_result[0], [-1]), -split_result[0]), axis=-1)
        if window_fn is not None:
            window = window_fn(2 * half_len, dtype=mdcts.dtype)
            real_frames *= window
        else:
            real_frames *= 1.0 / np.sqrt(2)
        return reconstruction_ops.overlap_and_add(real_frames, half_len)