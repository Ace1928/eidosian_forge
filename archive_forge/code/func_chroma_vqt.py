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
def chroma_vqt(*, y: Optional[np.ndarray]=None, sr: float=22050, V: Optional[np.ndarray]=None, hop_length: int=512, fmin: Optional[float]=None, intervals: Union[str, Collection[float]], norm: Optional[float]=np.inf, threshold: float=0.0, n_octaves: int=7, bins_per_octave: int=12, gamma: float=0) -> np.ndarray:
    """Variable-Q chromagram

    This differs from CQT-based chroma by supporting non-equal temperament
    intervals.

    Note: unlike CQT- and STFT-based chroma, VQT chroma does not aggregate energy
    from neighboring frequency bands.  As a result, the number of chroma
    features produced is equal to the number of intervals used, or equivalently,
    the number of bins per octave in the underlying VQT representation.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)]
        audio time series. Multi-channel is supported.
    sr : number > 0
        sampling rate of ``y``
    V : np.ndarray [shape=(..., d, t)] [Optional]
        a pre-computed variable-Q spectrogram
    hop_length : int > 0
        number of samples between successive chroma frames
    fmin : float > 0
        minimum frequency to analyze in the CQT.
        Default: `C1 ~= 32.7 Hz`
    intervals : str or array of floats in [1, 2)
        Either a string specification for an interval set, e.g.,
        `'equal'`, `'pythagorean'`, `'ji3'`, etc. or an array of
        intervals expressed as numbers between 1 and 2.
        .. see also:: librosa.interval_frequencies
    norm : int > 0, +-np.inf, or None
        Column-wise normalization of the chromagram.
    threshold : float
        Pre-normalization energy threshold.  Values below the
        threshold are discarded, resulting in a sparse chromagram.
    n_octaves : int > 0
        Number of octaves to analyze above ``fmin``
    bins_per_octave : int > 0, optional
        Number of bins per octave in the VQT.
    gamma : number > 0 [scalar]
        Bandwidth offset for determining filter lengths.
        .. see also:: librosa.vqt

    Returns
    -------
    chromagram : np.ndarray [shape=(..., bins_per_octave, t)]
        The output chromagram

    See Also
    --------
    librosa.util.normalize
    librosa.vqt
    chroma_cqt
    librosa.interval_frequencies

    Examples
    --------
    Compare an equal-temperament CQT chromagram to a 5-limit just intonation
    chromagram.  Both use 36 bins per octave.

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> n_bins = 36
    >>> chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=n_bins)
    >>> chroma_vq = librosa.feature.chroma_vqt(y=y, sr=sr,
    ...                                        intervals='ji5',
    ...                                        bins_per_octave=n_bins)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time',
    ...                          ax=ax[0], bins_per_octave=n_bins)
    >>> ax[0].set(ylabel='chroma_cqt')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(chroma_vq, y_axis='chroma_fjs', x_axis='time',
    ...                                ax=ax[1], bins_per_octave=n_bins,
    ...                                intervals='ji5')
    >>> ax[1].set(ylabel='chroma_vqt')
    >>> fig.colorbar(img, ax=ax)
    """
    if not isinstance(intervals, str):
        bins_per_octave = len(intervals)
    if V is None:
        if y is None:
            raise ParameterError('At least one of y or V must be provided to compute chroma')
        V = np.abs(vqt(y=y, sr=sr, hop_length=hop_length, fmin=fmin, intervals=intervals, n_bins=n_octaves * bins_per_octave, bins_per_octave=bins_per_octave, gamma=gamma))
    vq_to_chr = filters.cq_to_chroma(V.shape[-2], bins_per_octave=bins_per_octave, n_chroma=bins_per_octave, fmin=fmin)
    chroma = np.einsum('cf,...ft->...ct', vq_to_chr, V, optimize=True)
    if threshold is not None:
        chroma[chroma < threshold] = 0.0
    chroma = util.normalize(chroma, norm=norm, axis=-2)
    return chroma