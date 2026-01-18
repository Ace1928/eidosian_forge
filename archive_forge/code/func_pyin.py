import warnings
import numpy as np
import scipy
import numba
from .spectrum import _spectrogram
from . import convert
from .._cache import cache
from .. import util
from .. import sequence
from ..util.exceptions import ParameterError
from numpy.typing import ArrayLike
from typing import Any, Callable, Optional, Tuple, Union
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
def pyin(y: np.ndarray, *, fmin: float, fmax: float, sr: float=22050, frame_length: int=2048, win_length: Optional[int]=None, hop_length: Optional[int]=None, n_thresholds: int=100, beta_parameters: Tuple[float, float]=(2, 18), boltzmann_parameter: float=2, resolution: float=0.1, max_transition_rate: float=35.92, switch_prob: float=0.01, no_trough_prob: float=0.01, fill_na: Optional[float]=np.nan, center: bool=True, pad_mode: _PadMode='constant') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fundamental frequency (F0) estimation using probabilistic YIN (pYIN).

    pYIN [#]_ is a modificatin of the YIN algorithm [#]_ for fundamental frequency (F0) estimation.
    In the first step of pYIN, F0 candidates and their probabilities are computed using the YIN algorithm.
    In the second step, Viterbi decoding is used to estimate the most likely F0 sequence and voicing flags.

    .. [#] Mauch, Matthias, and Simon Dixon.
        "pYIN: A fundamental frequency estimator using probabilistic threshold distributions."
        2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.

    .. [#] De Cheveigné, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.
    fmin : number > 0 [scalar]
        minimum frequency in Hertz.
        The recommended minimum is ``librosa.note_to_hz('C2')`` (~65 Hz)
        though lower values may be feasible.
    fmax : number > fmin, <= sr/2 [scalar]
        maximum frequency in Hertz.
        The recommended maximum is ``librosa.note_to_hz('C7')`` (~2093 Hz)
        though higher values may be feasible.
    sr : number > 0 [scalar]
        sampling rate of ``y`` in Hertz.
    frame_length : int > 0 [scalar]
        length of the frames in samples.
        By default, ``frame_length=2048`` corresponds to a time scale of about 93 ms at
        a sampling rate of 22050 Hz.
    win_length : None or int > 0 [scalar]
        length of the window for calculating autocorrelation in samples.
        If ``None``, defaults to ``frame_length // 2``
    hop_length : None or int > 0 [scalar]
        number of audio samples between adjacent pYIN predictions.
        If ``None``, defaults to ``frame_length // 4``.
    n_thresholds : int > 0 [scalar]
        number of thresholds for peak estimation.
    beta_parameters : tuple
        shape parameters for the beta distribution prior over thresholds.
    boltzmann_parameter : number > 0 [scalar]
        shape parameter for the Boltzmann distribution prior over troughs.
        Larger values will assign more mass to smaller periods.
    resolution : float in `(0, 1)`
        Resolution of the pitch bins.
        0.01 corresponds to cents.
    max_transition_rate : float > 0
        maximum pitch transition rate in octaves per second.
    switch_prob : float in ``(0, 1)``
        probability of switching from voiced to unvoiced or vice versa.
    no_trough_prob : float in ``(0, 1)``
        maximum probability to add to global minimum if no trough is below threshold.
    fill_na : None, float, or ``np.nan``
        default value for unvoiced frames of ``f0``.
        If ``None``, the unvoiced frames will contain a best guess value.
    center : boolean
        If ``True``, the signal ``y`` is padded so that frame
        ``D[:, t]`` is centered at ``y[t * hop_length]``.
        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.
        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
        time grid by means of ``librosa.core.frames_to_samples``.
    pad_mode : string or function
        If ``center=True``, this argument is passed to ``np.pad`` for padding
        the edges of the signal ``y``. By default (``pad_mode="constant"``),
        ``y`` is padded on both sides with zeros.
        If ``center=False``,  this argument is ignored.
        .. see also:: `np.pad`

    Returns
    -------
    f0: np.ndarray [shape=(..., n_frames)]
        time series of fundamental frequencies in Hertz.
    voiced_flag: np.ndarray [shape=(..., n_frames)]
        time series containing boolean flags indicating whether a frame is voiced or not.
    voiced_prob: np.ndarray [shape=(..., n_frames)]
        time series containing the probability that a frame is voiced.
    .. note:: If multi-channel input is provided, f0 and voicing are estimated separately for each channel.

    See Also
    --------
    librosa.yin :
        Fundamental frequency (F0) estimation using the YIN algorithm.

    Examples
    --------
    Computing a fundamental frequency (F0) curve from an audio input

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> f0, voiced_flag, voiced_probs = librosa.pyin(y,
    ...                                              fmin=librosa.note_to_hz('C2'),
    ...                                              fmax=librosa.note_to_hz('C7'))
    >>> times = librosa.times_like(f0)

    Overlay F0 over a spectrogram

    >>> import matplotlib.pyplot as plt
    >>> D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    >>> ax.set(title='pYIN fundamental frequency estimation')
    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")
    >>> ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    >>> ax.legend(loc='upper right')
    """
    if fmin is None or fmax is None:
        raise ParameterError('both "fmin" and "fmax" must be provided')
    if win_length is None:
        win_length = frame_length // 2
    __check_yin_params(sr=sr, fmax=fmax, fmin=fmin, frame_length=frame_length, win_length=win_length)
    if hop_length is None:
        hop_length = frame_length // 4
    util.valid_audio(y, mono=False)
    if center:
        padding = [(0, 0) for _ in y.shape]
        padding[-1] = (frame_length // 2, frame_length // 2)
        y = np.pad(y, padding, mode=pad_mode)
    y_frames = util.frame(y, frame_length=frame_length, hop_length=hop_length)
    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)
    yin_frames = _cumulative_mean_normalized_difference(y_frames, frame_length, win_length, min_period, max_period)
    parabolic_shifts = _parabolic_interpolation(yin_frames)
    thresholds = np.linspace(0, 1, n_thresholds + 1)
    beta_cdf = scipy.stats.beta.cdf(thresholds, beta_parameters[0], beta_parameters[1])
    beta_probs = np.diff(beta_cdf)
    n_bins_per_semitone = int(np.ceil(1.0 / resolution))
    n_pitch_bins = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1

    def _helper(a, b):
        return __pyin_helper(a, b, sr, thresholds, boltzmann_parameter, beta_probs, no_trough_prob, min_period, fmin, n_pitch_bins, n_bins_per_semitone)
    helper = np.vectorize(_helper, signature='(f,t),(k,t)->(1,d,t),(j,t)')
    observation_probs, voiced_prob = helper(yin_frames, parabolic_shifts)
    max_semitones_per_frame = round(max_transition_rate * 12 * hop_length / sr)
    transition_width = max_semitones_per_frame * n_bins_per_semitone + 1
    transition = sequence.transition_local(n_pitch_bins, transition_width, window='triangle', wrap=False)
    t_switch = sequence.transition_loop(2, 1 - switch_prob)
    transition = np.kron(t_switch, transition)
    p_init = np.zeros(2 * n_pitch_bins)
    p_init[n_pitch_bins:] = 1 / n_pitch_bins
    states = sequence.viterbi(observation_probs, transition, p_init=p_init)
    freqs = fmin * 2 ** (np.arange(n_pitch_bins) / (12 * n_bins_per_semitone))
    f0 = freqs[states % n_pitch_bins]
    voiced_flag = states < n_pitch_bins
    if fill_na is not None:
        f0[~voiced_flag] = fill_na
    return (f0[..., 0, :], voiced_flag[..., 0, :], voiced_prob[..., 0, :])