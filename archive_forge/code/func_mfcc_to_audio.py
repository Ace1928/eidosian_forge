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
def mfcc_to_audio(mfcc: np.ndarray, *, n_mels: int=128, dct_type: int=2, norm: Optional[str]='ortho', ref: float=1.0, lifter: float=0, **kwargs: Any) -> np.ndarray:
    """Convert Mel-frequency cepstral coefficients to a time-domain audio signal

    This function is primarily a convenience wrapper for the following steps:

        1. Convert mfcc to Mel power spectrum (`mfcc_to_mel`)
        2. Convert Mel power spectrum to time-domain audio (`mel_to_audio`)

    Parameters
    ----------
    mfcc : np.ndarray [shape=(..., n_mfcc, n)]
        The Mel-frequency cepstral coefficients
    n_mels : int > 0
        The number of Mel frequencies
    dct_type : {1, 2, 3}
        Discrete cosine transform (DCT) type
        By default, DCT type-2 is used.
    norm : None or 'ortho'
        If ``dct_type`` is `2 or 3`, setting ``norm='ortho'`` uses an orthonormal
        DCT basis.
        Normalization is not supported for ``dct_type=1``.
    ref : float
        Reference power for (inverse) decibel calculation
    lifter : number >= 0
        If ``lifter>0``, apply inverse liftering (inverse cepstral filtering)::
            M[n, :] <- M[n, :] / (1 + sin(pi * (n + 1) / lifter)) * lifter / 2
    **kwargs : additional keyword arguments to pass through to `mel_to_audio`
    M : np.ndarray [shape=(..., n_mels, n), non-negative]
        The spectrogram as produced by `feature.melspectrogram`
    sr : number > 0 [scalar]
        sampling rate of the underlying signal
    n_fft : int > 0 [scalar]
        number of FFT components in the resulting STFT
    hop_length : None or int > 0
        The hop length of the STFT.  If not provided, it will default to ``n_fft // 4``
    win_length : None or int > 0
        The window length of the STFT.  By default, it will equal ``n_fft``
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        A window specification as supported by `stft` or `istft`
    center : boolean
        If `True`, the STFT is assumed to use centered frames.
        If `False`, the STFT is assumed to use left-aligned frames.
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram
    n_iter : int > 0
        The number of iterations for Griffin-Lim
    length : None or int > 0
        If provided, the output ``y`` is zero-padded or clipped to exactly ``length``
        samples.
    dtype : np.dtype
        Real numeric type for the time-domain signal.  Default is 32-bit float.
    **kwargs : additional keyword arguments for Mel filter bank parameters
    fmin : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    htk : bool [scalar]
        use HTK formula instead of Slaney

    Returns
    -------
    y : np.ndarray [shape=(..., n)]
        A time-domain signal reconstructed from `mfcc`

    See Also
    --------
    mfcc_to_mel
    mel_to_audio
    librosa.feature.mfcc
    librosa.griffinlim
    scipy.fftpack.dct
    """
    mel_spec = mfcc_to_mel(mfcc, n_mels=n_mels, dct_type=dct_type, norm=norm, ref=ref, lifter=lifter)
    return mel_to_audio(mel_spec, **kwargs)