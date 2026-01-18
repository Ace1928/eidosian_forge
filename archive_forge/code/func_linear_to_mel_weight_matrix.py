from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('signal.linear_to_mel_weight_matrix')
@dispatch.add_dispatch_support
def linear_to_mel_weight_matrix(num_mel_bins=20, num_spectrogram_bins=129, sample_rate=8000, lower_edge_hertz=125.0, upper_edge_hertz=3800.0, dtype=dtypes.float32, name=None):
    """Returns a matrix to warp linear scale spectrograms to the [mel scale][mel].

  Returns a weight matrix that can be used to re-weight a `Tensor` containing
  `num_spectrogram_bins` linearly sampled frequency information from
  `[0, sample_rate / 2]` into `num_mel_bins` frequency information from
  `[lower_edge_hertz, upper_edge_hertz]` on the [mel scale][mel].

  This function follows the [Hidden Markov Model Toolkit
  (HTK)](http://htk.eng.cam.ac.uk/) convention, defining the mel scale in
  terms of a frequency in hertz according to the following formula:

      $$\\textrm{mel}(f) = 2595 * \\textrm{log}_{10}(1 + \\frac{f}{700})$$

  In the returned matrix, all the triangles (filterbanks) have a peak value
  of 1.0.

  For example, the returned matrix `A` can be used to right-multiply a
  spectrogram `S` of shape `[frames, num_spectrogram_bins]` of linear
  scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram"
  `M` of shape `[frames, num_mel_bins]`.

      # `S` has shape [frames, num_spectrogram_bins]
      # `M` has shape [frames, num_mel_bins]
      M = tf.matmul(S, A)

  The matrix can be used with `tf.tensordot` to convert an arbitrary rank
  `Tensor` of linear-scale spectral bins into the mel scale.

      # S has shape [..., num_spectrogram_bins].
      # M has shape [..., num_mel_bins].
      M = tf.tensordot(S, A, 1)

  Args:
    num_mel_bins: Python int. How many bands in the resulting mel spectrum.
    num_spectrogram_bins: An integer `Tensor`. How many bins there are in the
      source spectrogram data, which is understood to be `fft_size // 2 + 1`,
      i.e. the spectrogram only contains the nonredundant FFT bins.
    sample_rate: An integer or float `Tensor`. Samples per second of the input
      signal used to create the spectrogram. Used to figure out the frequencies
      corresponding to each spectrogram bin, which dictates how they are mapped
      into the mel scale.
    lower_edge_hertz: Python float. Lower bound on the frequencies to be
      included in the mel spectrum. This corresponds to the lower edge of the
      lowest triangular band.
    upper_edge_hertz: Python float. The desired top edge of the highest
      frequency band.
    dtype: The `DType` of the result matrix. Must be a floating point type.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of shape `[num_spectrogram_bins, num_mel_bins]`.

  Raises:
    ValueError: If `num_mel_bins`/`num_spectrogram_bins`/`sample_rate` are not
      positive, `lower_edge_hertz` is negative, frequency edges are incorrectly
      ordered, `upper_edge_hertz` is larger than the Nyquist frequency.

  [mel]: https://en.wikipedia.org/wiki/Mel_scale
  """
    with ops.name_scope(name, 'linear_to_mel_weight_matrix') as name:
        if isinstance(sample_rate, tensor.Tensor):
            maybe_const_val = tensor_util.constant_value(sample_rate)
            if maybe_const_val is not None:
                sample_rate = maybe_const_val
        _validate_arguments(num_mel_bins, sample_rate, lower_edge_hertz, upper_edge_hertz, dtype)
        sample_rate = math_ops.cast(sample_rate, dtype, name='sample_rate')
        lower_edge_hertz = ops.convert_to_tensor(lower_edge_hertz, dtype, name='lower_edge_hertz')
        upper_edge_hertz = ops.convert_to_tensor(upper_edge_hertz, dtype, name='upper_edge_hertz')
        zero = ops.convert_to_tensor(0.0, dtype)
        bands_to_zero = 1
        nyquist_hertz = sample_rate / 2.0
        linear_frequencies = math_ops.linspace(zero, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:]
        spectrogram_bins_mel = array_ops.expand_dims(_hertz_to_mel(linear_frequencies), 1)
        band_edges_mel = shape_ops.frame(math_ops.linspace(_hertz_to_mel(lower_edge_hertz), _hertz_to_mel(upper_edge_hertz), num_mel_bins + 2), frame_length=3, frame_step=1)
        lower_edge_mel, center_mel, upper_edge_mel = tuple((array_ops.reshape(t, [1, num_mel_bins]) for t in array_ops.split(band_edges_mel, 3, axis=1)))
        lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
        upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
        mel_weights_matrix = math_ops.maximum(zero, math_ops.minimum(lower_slopes, upper_slopes))
        return array_ops.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]], name=name)