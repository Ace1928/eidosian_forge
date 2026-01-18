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
@cache(level=20)
def iirt(y: np.ndarray, *, sr: float=22050, win_length: int=2048, hop_length: Optional[int]=None, center: bool=True, tuning: float=0.0, pad_mode: _PadMode='constant', flayout: str='sos', res_type: str='soxr_hq', **kwargs: Any) -> np.ndarray:
    """Time-frequency representation using IIR filters

    This function will return a time-frequency representation
    using a multirate filter bank consisting of IIR filters. [#]_

    First, ``y`` is resampled as needed according to the provided ``sample_rates``.

    Then, a filterbank with with ``n`` band-pass filters is designed.

    The resampled input signals are processed by the filterbank as a whole.
    (`scipy.signal.filtfilt` resp. `sosfiltfilt` is used to make the phase linear.)
    The output of the filterbank is cut into frames.
    For each band, the short-time mean-square power (STMSP) is calculated by
    summing ``win_length`` subsequent filtered time samples.

    When called with the default set of parameters, it will generate the TF-representation
    (pitch filterbank):

        * 85 filters with MIDI pitches [24, 108] as ``center_freqs``.
        * each filter having a bandwidth of one semitone.

    .. [#] Müller, Meinard.
           "Information Retrieval for Music and Motion."
           Springer Verlag. 2007.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    win_length : int > 0, <= n_fft
        Window length.
    hop_length : int > 0 [scalar]
        Hop length, number samples between subsequent frames.
        If not supplied, defaults to ``win_length // 4``.
    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``D[..., :, t]`` is centered at ``y[t * hop_length]``.
        - If ``False``, then `D[..., :, t]`` begins at ``y[t * hop_length]``
    tuning : float [scalar]
        Tuning deviation from A440 in fractions of a bin.
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, this function uses zero padding.
    flayout : string
        - If `sos` (default), a series of second-order filters is used for filtering with `scipy.signal.sosfiltfilt`.
          Minimizes numerical precision errors for high-order filters, but is slower.
        - If `ba`, the standard difference equation is used for filtering with `scipy.signal.filtfilt`.
          Can be unstable for high-order filters.
    res_type : string
        The resampling mode.  See `librosa.resample` for details.
    **kwargs : additional keyword arguments
        Additional arguments for `librosa.filters.semitone_filterbank`
        (e.g., could be used to provide another set of ``center_freqs`` and ``sample_rates``).

    Returns
    -------
    bands_power : np.ndarray [shape=(..., n, t), dtype=dtype]
        Short-time mean-square power for the input signal.

    Raises
    ------
    ParameterError
        If ``flayout`` is not None, `ba`, or `sos`.

    See Also
    --------
    librosa.filters.semitone_filterbank
    librosa.filters.mr_frequencies
    librosa.cqt
    scipy.signal.filtfilt
    scipy.signal.sosfiltfilt

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> D = np.abs(librosa.iirt(y))
    >>> C = np.abs(librosa.cqt(y=y, sr=sr))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
    ...                                y_axis='cqt_hz', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Constant-Q transform')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                                y_axis='cqt_hz', x_axis='time', ax=ax[1])
    >>> ax[1].set_title('Semitone spectrogram (iirt)')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """
    if flayout not in ('ba', 'sos'):
        raise ParameterError(f'Unsupported flayout={flayout}')
    util.valid_audio(y, mono=False)
    if hop_length is None:
        hop_length = win_length // 4
    if center:
        padding = [(0, 0) for _ in y.shape]
        padding[-1] = (win_length // 2, win_length // 2)
        y = np.pad(y, padding, mode=pad_mode)
    filterbank_ct, sample_rates = semitone_filterbank(tuning=tuning, flayout=flayout, **kwargs)
    y_resampled = []
    y_srs = np.unique(sample_rates)
    for cur_sr in y_srs:
        y_resampled.append(resample(y, orig_sr=sr, target_sr=cur_sr, res_type=res_type))
    n_frames = int(1 + (y.shape[-1] - win_length) // hop_length)
    shape = list(y.shape)
    shape[-1] = n_frames
    shape.insert(-1, len(filterbank_ct))
    bands_power = np.empty_like(y, shape=shape)
    slices: List[Union[int, slice]] = [slice(None) for _ in bands_power.shape]
    for i, (cur_sr, cur_filter) in enumerate(zip(sample_rates, filterbank_ct)):
        slices[-2] = i
        cur_sr_idx = np.flatnonzero(y_srs == cur_sr)[0]
        if flayout == 'ba':
            cur_filter_output = scipy.signal.filtfilt(cur_filter[0], cur_filter[1], y_resampled[cur_sr_idx], axis=-1)
        elif flayout == 'sos':
            cur_filter_output = scipy.signal.sosfiltfilt(cur_filter, y_resampled[cur_sr_idx], axis=-1)
        factor = sr / cur_sr
        hop_length_STMSP = hop_length / factor
        win_length_STMSP_round = int(round(win_length / factor))
        start_idx = np.arange(0, cur_filter_output.shape[-1] - win_length_STMSP_round, hop_length_STMSP)
        if len(start_idx) < n_frames:
            min_length = int(np.ceil(n_frames * hop_length_STMSP)) + win_length_STMSP_round
            cur_filter_output = util.fix_length(cur_filter_output, size=min_length)
            start_idx = np.arange(0, cur_filter_output.shape[-1] - win_length_STMSP_round, hop_length_STMSP)
        start_idx = np.round(start_idx).astype(int)[:n_frames]
        idx = np.add.outer(start_idx, np.arange(win_length_STMSP_round))
        bands_power[tuple(slices)] = factor * np.sum(cur_filter_output[..., idx] ** 2, axis=-1)
    return bands_power