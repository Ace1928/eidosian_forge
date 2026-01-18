from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def multi_frequency_weighting(frequencies: _ScalarOrSequence[_FloatLike_co], *, kinds: Iterable[str]='ZAC', **kwargs: Any) -> np.ndarray:
    """Compute multiple weightings of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    kinds : list or tuple or str
        An iterable of weighting kinds. e.g. `('Z', 'B')`, `'ZAD'`, `'C'`
    **kwargs : keywords to pass to the weighting function.

    Returns
    -------
    weighting : scalar or np.ndarray [shape=(len(kinds), n)]
        ``weighting[i, j]`` is the weighting of ``frequencies[j]``
        using the curve determined by ``kinds[i]``.

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    A_weighting
    B_weighting
    C_weighting
    D_weighting

    Examples
    --------
    Get the A, B, C, D, and Z weightings for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weightings = 'ABCDZ'
    >>> weights = librosa.multi_frequency_weighting(freqs, kinds=weightings)
    >>> fig, ax = plt.subplots()
    >>> for label, w in zip(weightings, weights):
    ...     ax.plot(freqs, w, label=label)
    >>> ax.set(xlabel='Frequency (Hz)', ylabel='Weighting (log10)',
    ...        title='Weightings of CQT frequencies')
    >>> ax.legend()
    """
    return np.stack([frequency_weighting(frequencies, kind=k, **kwargs) for k in kinds], axis=0)