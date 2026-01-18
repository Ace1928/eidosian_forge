import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
from .. import util
from .. import filters
from ..util.exceptions import ParameterError
from ..core.convert import fft_frequencies
from ..core.audio import zero_crossings
from ..core.spectrum import power_to_db, _spectrogram
from ..core.constantq import cqt, hybrid_cqt, vqt
from ..core.pitch import estimate_tuning
from typing import Any, Optional, Union, Collection
from numpy.typing import DTypeLike
from .._typing import _FloatLike_co, _WindowSpec, _PadMode, _PadModeSTFT
def chroma_cqt(*, y: Optional[np.ndarray]=None, sr: float=22050, C: Optional[np.ndarray]=None, hop_length: int=512, fmin: Optional[_FloatLike_co]=None, norm: Optional[Union[int, float]]=np.inf, threshold: float=0.0, tuning: Optional[float]=None, n_chroma: int=12, n_octaves: int=7, window: Optional[np.ndarray]=None, bins_per_octave: Optional[int]=36, cqt_mode: str='full') -> np.ndarray:
    """Constant-Q chromagram

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)]
        audio time series. Multi-channel is supported.
    sr : number > 0
        sampling rate of ``y``
    C : np.ndarray [shape=(..., d, t)] [Optional]
        a pre-computed constant-Q spectrogram
    hop_length : int > 0
        number of samples between successive chroma frames
    fmin : float > 0
        minimum frequency to analyze in the CQT.
        Default: `C1 ~= 32.7 Hz`
    norm : int > 0, +-np.inf, or None
        Column-wise normalization of the chromagram.
    threshold : float
        Pre-normalization energy threshold.  Values below the
        threshold are discarded, resulting in a sparse chromagram.
    tuning : float [scalar] or None.
        Deviation (in fractions of a CQT bin) from A440 tuning
    n_chroma : int > 0
        Number of chroma bins to produce
    n_octaves : int > 0
        Number of octaves to analyze above ``fmin``
    window : None or np.ndarray
        Optional window parameter to `filters.cq_to_chroma`
    bins_per_octave : int > 0, optional
        Number of bins per octave in the CQT.
        Must be an integer multiple of ``n_chroma``.
        Default: 36 (3 bins per semitone)
        If `None`, it will match ``n_chroma``.
    cqt_mode : ['full', 'hybrid']
        Constant-Q transform mode

    Returns
    -------
    chromagram : np.ndarray [shape=(..., n_chroma, t)]
        The output chromagram

    See Also
    --------
    librosa.util.normalize
    librosa.cqt
    librosa.hybrid_cqt
    chroma_stft

    Examples
    --------
    Compare a long-window STFT chromagram to the CQT chromagram

    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=15)
    >>> chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr,
    ...                                           n_chroma=12, n_fft=4096)
    >>> chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='chroma_stft')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='chroma_cqt')
    >>> fig.colorbar(img, ax=ax)
    """
    cqt_func = {'full': cqt, 'hybrid': hybrid_cqt}
    if bins_per_octave is None:
        bins_per_octave = n_chroma
    elif np.remainder(bins_per_octave, n_chroma) != 0:
        raise ParameterError(f'bins_per_octave={bins_per_octave} must be an integer multiple of n_chroma={n_chroma}')
    if C is None:
        if y is None:
            raise ParameterError('At least one of C or y must be provided to compute chroma')
        C = np.abs(cqt_func[cqt_mode](y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_octaves * bins_per_octave, bins_per_octave=bins_per_octave, tuning=tuning))
    cq_to_chr = filters.cq_to_chroma(C.shape[-2], bins_per_octave=bins_per_octave, n_chroma=n_chroma, fmin=fmin, window=window)
    chroma = np.einsum('cf,...ft->...ct', cq_to_chr, C, optimize=True)
    if threshold is not None:
        chroma[chroma < threshold] = 0.0
    chroma = util.normalize(chroma, norm=norm, axis=-2)
    return chroma