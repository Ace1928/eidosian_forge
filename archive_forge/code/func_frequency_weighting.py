from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def frequency_weighting(frequencies: _ScalarOrSequence[_FloatLike_co], *, kind: str='A', **kwargs: Any) -> Union[np.floating[Any], np.ndarray]:
    """Compute the weighting of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    kind : str in
        The weighting kind. e.g. `'A'`, `'B'`, `'C'`, `'D'`, `'Z'`
    **kwargs
        Additional keyword arguments to A_weighting, B_weighting, etc.

    Returns
    -------
    weighting : scalar or np.ndarray [shape=(n,)]
        ``weighting[i]`` is the weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    multi_frequency_weighting
    A_weighting
    B_weighting
    C_weighting
    D_weighting

    Examples
    --------
    Get the A-weighting for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weights = librosa.frequency_weighting(freqs, kind='A')
    >>> fig, ax = plt.subplots()
    >>> ax.plot(freqs, weights)
    >>> ax.set(xlabel='Frequency (Hz)', ylabel='Weighting (log10)',
    ...        title='A-Weighting of CQT frequencies')
    """
    if isinstance(kind, str):
        kind = kind.upper()
    return WEIGHTING_FUNCTIONS[kind](frequencies, **kwargs)