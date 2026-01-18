from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@classmethod
def from_window(cls, win_param: str | tuple | float, fs: float, nperseg: int, noverlap: int, *, symmetric_win: bool=False, fft_mode: FFT_MODE_TYPE='onesided', mfft: int | None=None, scale_to: Literal['magnitude', 'psd'] | None=None, phase_shift: int | None=0):
    """Instantiate `ShortTimeFFT` by using `get_window`.

        The method `get_window` is used to create a window of length
        `nperseg`. The parameter names `noverlap`, and `nperseg` are used here,
        since they more inline with other classical STFT libraries.

        Parameters
        ----------
        win_param: Union[str, tuple, float],
            Parameters passed to `get_window`. For windows with no parameters,
            it may be a string (e.g., ``'hann'``), for parametrized windows a
            tuple, (e.g., ``('gaussian', 2.)``) or a single float specifying
            the shape parameter of a kaiser window (i.e. ``4.``  and
            ``('kaiser', 4.)`` are equal. See `get_window` for more details.
        fs : float
            Sampling frequency of input signal. Its relation to the
            sampling interval `T` is ``T = 1 / fs``.
        nperseg: int
            Window length in samples, which corresponds to the `m_num`.
        noverlap: int
            Window overlap in samples. It relates to the `hop` increment by
            ``hop = npsereg - noverlap``.
        symmetric_win: bool
            If ``True`` then a symmetric window is generated, else a periodic
            window is generated (default). Though symmetric windows seem for
            most applications to be more sensible, the default of a periodic
            windows was chosen to correspond to the default of `get_window`.
        fft_mode : 'twosided', 'centered', 'onesided', 'onesided2X'
            Mode of FFT to be used (default 'onesided').
            See property `fft_mode` for details.
        mfft: int | None
            Length of the FFT used, if a zero padded FFT is desired.
            If ``None`` (default), the length of the window `win` is used.
        scale_to : 'magnitude', 'psd' | None
            If not ``None`` (default) the window function is scaled, so each
            STFT column represents  either a 'magnitude' or a power spectral
            density ('psd') spectrum. This parameter sets the property
            `scaling` to the same value. See method `scale_to` for details.
        phase_shift : int | None
            If set, add a linear phase `phase_shift` / `mfft` * `f` to each
            frequency `f`. The default value 0 ensures that there is no phase
            shift on the zeroth slice (in which t=0 is centered). See property
            `phase_shift` for more details.

        Examples
        --------
        The following instances ``SFT0`` and ``SFT1`` are equivalent:

        >>> from scipy.signal import ShortTimeFFT, get_window
        >>> nperseg = 9  # window length
        >>> w = get_window(('gaussian', 2.), nperseg)
        >>> fs = 128  # sampling frequency
        >>> hop = 3  # increment of STFT time slice
        >>> SFT0 = ShortTimeFFT(w, hop, fs=fs)
        >>> SFT1 = ShortTimeFFT.from_window(('gaussian', 2.), fs, nperseg,
        ...                                 noverlap=nperseg-hop)

        See Also
        --------
        scipy.signal.get_window: Return a window of a given length and type.
        from_dual: Create instance using dual window.
        ShortTimeFFT: Create instance using standard initializer.
        """
    win = get_window(win_param, nperseg, fftbins=not symmetric_win)
    return cls(win, hop=nperseg - noverlap, fs=fs, fft_mode=fft_mode, mfft=mfft, scale_to=scale_to, phase_shift=phase_shift)