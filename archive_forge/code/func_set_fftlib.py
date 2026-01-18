from types import ModuleType
from typing import Optional
def set_fftlib(lib: Optional[ModuleType]=None) -> None:
    """Set the FFT library used by librosa.

    Parameters
    ----------
    lib : None or module
        Must implement an interface compatible with `numpy.fft`.
        If ``None``, reverts to `numpy.fft`.

    Examples
    --------
    Use `pyfftw`:

    >>> import pyfftw
    >>> librosa.set_fftlib(pyfftw.interfaces.numpy_fft)

    Reset to default `numpy` implementation

    >>> librosa.set_fftlib()
    """
    global __FFTLIB
    if lib is None:
        from numpy import fft
        lib = fft
    __FFTLIB = lib