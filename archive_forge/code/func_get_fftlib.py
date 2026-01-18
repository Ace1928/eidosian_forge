from types import ModuleType
from typing import Optional
def get_fftlib() -> ModuleType:
    """Get the FFT library currently used by librosa

    Returns
    -------
    fft : module
        The FFT library currently used by librosa.
        Must API-compatible with `numpy.fft`.
    """
    if __FFTLIB is None:
        assert False
    return __FFTLIB