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
def db_to_amplitude(S_db: np.ndarray, *, ref: float=1.0) -> np.ndarray:
    """Convert a dB-scaled spectrogram to an amplitude spectrogram.

    This effectively inverts `amplitude_to_db`::

        db_to_amplitude(S_db) ~= 10.0**(0.5 * S_db/10 + log10(ref))

    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled spectrogram
    ref : number > 0
        Optional reference power.

    Returns
    -------
    S : np.ndarray
        Linear magnitude spectrogram

    Notes
    -----
    This function caches at level 30.
    """
    return db_to_power(S_db, ref=ref ** 2) ** 0.5