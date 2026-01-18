from numbers import Number
import operator
import os
import threading
import contextlib
import numpy as np
from .pypocketfft import good_size
@contextlib.contextmanager
def set_workers(workers):
    """Context manager for the default number of workers used in `scipy.fft`

    Parameters
    ----------
    workers : int
        The default number of workers to use

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fft, signal
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal((128, 64))
    >>> with fft.set_workers(4):
    ...     y = signal.fftconvolve(x, x)

    """
    old_workers = get_workers()
    _config.default_workers = _workers(operator.index(workers))
    try:
        yield
    finally:
        _config.default_workers = old_workers