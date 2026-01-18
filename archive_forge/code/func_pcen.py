import warnings
import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import scipy.interpolate
from numba import jit
from . import convert
from .fft import get_fftlib
from .audio import resample
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..filters import get_window, semitone_filterbank
from ..filters import window_sumsquare
from numpy.typing import DTypeLike
from typing import Any, Callable, Optional, Tuple, List, Union, overload
from typing_extensions import Literal
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
@cache(level=30)
def pcen(S: np.ndarray, *, sr: float=22050, hop_length: int=512, gain: float=0.98, bias: float=2, power: float=0.5, time_constant: float=0.4, eps: float=1e-06, b: Optional[float]=None, max_size: int=1, ref: Optional[np.ndarray]=None, axis: int=-1, max_axis: Optional[int]=None, zi: Optional[np.ndarray]=None, return_zf: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Per-channel energy normalization (PCEN)

    This function normalizes a time-frequency representation ``S`` by
    performing automatic gain control, followed by nonlinear compression [#]_ ::

        P[f, t] = (S / (eps + M[f, t])**gain + bias)**power - bias**power

    IMPORTANT: the default values of eps, gain, bias, and power match the
    original publication, in which ``S`` is a 40-band mel-frequency
    spectrogram with 25 ms windowing, 10 ms frame shift, and raw audio values
    in the interval [-2**31; 2**31-1[. If you use these default values, we
    recommend to make sure that the raw audio is properly scaled to this
    interval, and not to [-1, 1[ as is most often the case.

    The matrix ``M`` is the result of applying a low-pass, temporal IIR filter
    to ``S``::

        M[f, t] = (1 - b) * M[f, t - 1] + b * S[f, t]

    If ``b`` is not provided, it is calculated as::

        b = (sqrt(1 + 4* T**2) - 1) / (2 * T**2)

    where ``T = time_constant * sr / hop_length``. [#]_

    This normalization is designed to suppress background noise and
    emphasize foreground signals, and can be used as an alternative to
    decibel scaling (`amplitude_to_db`).

    This implementation also supports smoothing across frequency bins
    by specifying ``max_size > 1``.  If this option is used, the filtered
    spectrogram ``M`` is computed as::

        M[f, t] = (1 - b) * M[f, t - 1] + b * R[f, t]

    where ``R`` has been max-filtered along the frequency axis, similar to
    the SuperFlux algorithm implemented in `onset.onset_strength`::

        R[f, t] = max(S[f - max_size//2: f + max_size//2, t])

    This can be used to perform automatic gain control on signals that cross
    or span multiple frequency bans, which may be desirable for spectrograms
    with high frequency resolution.

    .. [#] Wang, Y., Getreuer, P., Hughes, T., Lyon, R. F., & Saurous, R. A.
       (2017, March). Trainable frontend for robust and far-field keyword spotting.
       In Acoustics, Speech and Signal Processing (ICASSP), 2017
       IEEE International Conference on (pp. 5670-5674). IEEE.

    .. [#] Lostanlen, V., Salamon, J., McFee, B., Cartwright, M., Farnsworth, A.,
       Kelling, S., and Bello, J. P. Per-Channel Energy Normalization: Why and How.
       IEEE Signal Processing Letters, 26(1), 39-43.

    Parameters
    ----------
    S : np.ndarray (non-negative)
        The input (magnitude) spectrogram

    sr : number > 0 [scalar]
        The audio sampling rate

    hop_length : int > 0 [scalar]
        The hop length of ``S``, expressed in samples

    gain : number >= 0 [scalar]
        The gain factor.  Typical values should be slightly less than 1.

    bias : number >= 0 [scalar]
        The bias point of the nonlinear compression (default: 2)

    power : number >= 0 [scalar]
        The compression exponent.  Typical values should be between 0 and 0.5.
        Smaller values of ``power`` result in stronger compression.
        At the limit ``power=0``, polynomial compression becomes logarithmic.

    time_constant : number > 0 [scalar]
        The time constant for IIR filtering, measured in seconds.

    eps : number > 0 [scalar]
        A small constant used to ensure numerical stability of the filter.

    b : number in [0, 1]  [scalar]
        The filter coefficient for the low-pass filter.
        If not provided, it will be inferred from ``time_constant``.

    max_size : int > 0 [scalar]
        The width of the max filter applied to the frequency axis.
        If left as `1`, no filtering is performed.

    ref : None or np.ndarray (shape=S.shape)
        An optional pre-computed reference spectrum (``R`` in the above).
        If not provided it will be computed from ``S``.

    axis : int [scalar]
        The (time) axis of the input spectrogram.

    max_axis : None or int [scalar]
        The frequency axis of the input spectrogram.
        If `None`, and ``S`` is two-dimensional, it will be inferred
        as the opposite from ``axis``.
        If ``S`` is not two-dimensional, and ``max_size > 1``, an error
        will be raised.

    zi : np.ndarray
        The initial filter delay values.

        This may be the ``zf`` (final delay values) of a previous call to ``pcen``, or
        computed by `scipy.signal.lfilter_zi`.

    return_zf : bool
        If ``True``, return the final filter delay values along with the PCEN output ``P``.
        This is primarily useful in streaming contexts, where the final state of one
        block of processing should be used to initialize the next block.

        If ``False`` (default) only the PCEN values ``P`` are returned.

    Returns
    -------
    P : np.ndarray, non-negative [shape=(n, m)]
        The per-channel energy normalized version of ``S``.
    zf : np.ndarray (optional)
        The final filter delay values.  Only returned if ``return_zf=True``.

    See Also
    --------
    amplitude_to_db
    librosa.onset.onset_strength

    Examples
    --------
    Compare PCEN to log amplitude (dB) scaling on Mel spectra

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('robin'))

    >>> # We recommend scaling y to the range [-2**31, 2**31[ before applying
    >>> # PCEN's default parameters. Furthermore, we use power=1 to get a
    >>> # magnitude spectrum instead of a power spectrum.
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, power=1)
    >>> log_S = librosa.amplitude_to_db(S, ref=np.max)
    >>> pcen_S = librosa.pcen(S * (2**31))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(log_S, x_axis='time', y_axis='mel', ax=ax[0])
    >>> ax[0].set(title='log amplitude (dB)', xlabel=None)
    >>> ax[0].label_outer()
    >>> imgpcen = librosa.display.specshow(pcen_S, x_axis='time', y_axis='mel', ax=ax[1])
    >>> ax[1].set(title='Per-channel energy normalization')
    >>> fig.colorbar(img, ax=ax[0], format="%+2.0f dB")
    >>> fig.colorbar(imgpcen, ax=ax[1])

    Compare PCEN with and without max-filtering

    >>> pcen_max = librosa.pcen(S * (2**31), max_size=3)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(pcen_S, x_axis='time', y_axis='mel', ax=ax[0])
    >>> ax[0].set(title='Per-channel energy normalization (no max-filter)')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(pcen_max, x_axis='time', y_axis='mel', ax=ax[1])
    >>> ax[1].set(title='Per-channel energy normalization (max_size=3)')
    >>> fig.colorbar(img, ax=ax)
    """
    if power < 0:
        raise ParameterError(f'power={power} must be nonnegative')
    if gain < 0:
        raise ParameterError(f'gain={gain} must be non-negative')
    if bias < 0:
        raise ParameterError(f'bias={bias} must be non-negative')
    if eps <= 0:
        raise ParameterError(f'eps={eps} must be strictly positive')
    if time_constant <= 0:
        raise ParameterError(f'time_constant={time_constant} must be strictly positive')
    if not util.is_positive_int(max_size):
        raise ParameterError(f'max_size={max_size} must be a positive integer')
    if b is None:
        t_frames = time_constant * sr / float(hop_length)
        b = (np.sqrt(1 + 4 * t_frames ** 2) - 1) / (2 * t_frames ** 2)
    if not 0 <= b <= 1:
        raise ParameterError(f'b={b} must be between 0 and 1')
    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('pcen was called on complex input so phase information will be discarded. To suppress this warning, call pcen(np.abs(D)) instead.', stacklevel=2)
        S = np.abs(S)
    if ref is None:
        if max_size == 1:
            ref = S
        elif S.ndim == 1:
            raise ParameterError('Max-filtering cannot be applied to 1-dimensional input')
        else:
            if max_axis is None:
                if S.ndim != 2:
                    raise ParameterError(f'Max-filtering a {S.ndim:d}-dimensional spectrogram requires you to specify max_axis')
                max_axis = np.mod(1 - axis, 2)
            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=max_axis)
    if zi is None:
        shape = tuple([1] * ref.ndim)
        zi = np.empty(shape)
        zi[:] = scipy.signal.lfilter_zi([b], [1, b - 1])[:]
    S_smooth: np.ndarray
    zf: np.ndarray
    S_smooth, zf = scipy.signal.lfilter([b], [1, b - 1], ref, zi=zi, axis=axis)
    smooth = np.exp(-gain * (np.log(eps) + np.log1p(S_smooth / eps)))
    S_out: np.ndarray
    if power == 0:
        S_out = np.log1p(S * smooth)
    elif bias == 0:
        S_out = np.exp(power * (np.log(S) + np.log(smooth)))
    else:
        S_out = bias ** power * np.expm1(power * np.log1p(S * smooth / bias))
    if return_zf:
        return (S_out, zf)
    else:
        return S_out