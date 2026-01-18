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
def mfcc_to_mel(mfcc: np.ndarray, *, n_mels: int=128, dct_type: int=2, norm: Optional[str]='ortho', ref: float=1.0, lifter: float=0) -> np.ndarray:
    """Invert Mel-frequency cepstral coefficients to approximate a Mel power
    spectrogram.

    This inversion proceeds in two steps:

        1. The inverse DCT is applied to the MFCCs
        2. `librosa.db_to_power` is applied to map the dB-scaled result to a power spectrogram

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
        Normalization is not supported for `dct_type=1`.
    ref : float
        Reference power for (inverse) decibel calculation
    lifter : number >= 0
        If ``lifter>0``, apply inverse liftering (inverse cepstral filtering)::
            M[n, :] <- M[n, :] / (1 + sin(pi * (n + 1) / lifter) * lifter / 2)

    Returns
    -------
    M : np.ndarray [shape=(..., n_mels, n)]
        An approximate Mel power spectrum recovered from ``mfcc``

    Warns
    -----
    UserWarning
        due to critical values in lifter array that invokes underflow.

    See Also
    --------
    librosa.feature.mfcc
    librosa.feature.melspectrogram
    scipy.fftpack.dct
    """
    if lifter > 0:
        n_mfcc = mfcc.shape[-2]
        idx = np.arange(1, 1 + n_mfcc, dtype=mfcc.dtype)
        idx = expand_to(idx, ndim=mfcc.ndim, axes=-2)
        lifter_sine = 1 + lifter * 0.5 * np.sin(np.pi * idx / lifter)
        if np.any(np.abs(lifter_sine) < np.finfo(lifter_sine.dtype).eps):
            warnings.warn(message='lifter array includes critical values that may invoke underflow.', category=UserWarning, stacklevel=2)
        mfcc = mfcc / (lifter_sine + tiny(mfcc))
    elif lifter != 0:
        raise ParameterError('MFCC to mel lifter must be a non-negative number.')
    logmel = scipy.fftpack.idct(mfcc, axis=-2, type=dct_type, norm=norm, n=n_mels)
    return db_to_power(logmel, ref=ref)