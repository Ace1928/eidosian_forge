import warnings
import numpy as np
import scipy.fftpack
from ..util.exceptions import ParameterError
from ..core.spectrum import griffinlim
from ..core.spectrum import db_to_power
from ..util.utils import tiny
from .. import filters
from ..util import nnls, expand_to
from numpy.typing import DTypeLike
from typing import Any, Callable, Optional, Union
from .._typing import _WindowSpec, _PadModeSTFT
def mel_to_stft(M: np.ndarray, *, sr: float=22050, n_fft: int=2048, power: float=2.0, **kwargs: Any) -> np.ndarray:
    """Approximate STFT magnitude from a Mel power spectrogram.

    Parameters
    ----------
    M : np.ndarray [shape=(..., n_mels, n), non-negative]
        The spectrogram as produced by `feature.melspectrogram`
    sr : number > 0 [scalar]
        sampling rate of the underlying signal
    n_fft : int > 0 [scalar]
        number of FFT components in the resulting STFT
    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram
    **kwargs : additional keyword arguments for Mel filter bank parameters
    fmin : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    htk : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 'slaney', or number} [scalar]
        If 'slaney', divide the triangular mel weights by the width of
        the mel band (area normalization).
        If numeric, use `librosa.util.normalize` to normalize each filter
        by to unit l_p norm. See `librosa.util.normalize` for a full
        description of supported norm values (including `+-np.inf`).
        Otherwise, leave all the triangles aiming for a peak value of 1.0
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    S : np.ndarray [shape=(..., n_fft, t), non-negative]
        An approximate linear magnitude spectrogram

    See Also
    --------
    librosa.feature.melspectrogram
    librosa.stft
    librosa.filters.mel
    librosa.util.nnls

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = librosa.util.abs2(librosa.stft(y))
    >>> mel_spec = librosa.feature.melspectrogram(S=S, sr=sr)
    >>> S_inv = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr)

    Compare the results visually

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max, top_db=None),
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Original STFT')
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(S_inv, ref=np.max, top_db=None),
    ...                          y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Reconstructed STFT')
    >>> ax[1].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(np.abs(S_inv - S),
    ...                                                  ref=S.max(), top_db=None),
    ...                          vmax=0, y_axis='log', x_axis='time', cmap='magma', ax=ax[2])
    >>> ax[2].set(title='Residual error (dB)')
    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")
    """
    mel_basis = filters.mel(sr=sr, n_fft=n_fft, n_mels=M.shape[-2], dtype=M.dtype, **kwargs)
    inverse = nnls(mel_basis, M)
    return np.power(inverse, 1.0 / power, out=inverse)