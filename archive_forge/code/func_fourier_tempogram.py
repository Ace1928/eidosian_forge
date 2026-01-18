import numpy as np
import scipy
from .. import util
from .._cache import cache
from ..core.audio import autocorrelate
from ..core.spectrum import stft
from ..core.convert import tempo_frequencies, time_to_frames
from ..core.harmonic import f0_harmonics
from ..util.exceptions import ParameterError
from ..filters import get_window
from typing import Optional, Callable, Any
from .._typing import _WindowSpec
def fourier_tempogram(*, y: Optional[np.ndarray]=None, sr: float=22050, onset_envelope: Optional[np.ndarray]=None, hop_length: int=512, win_length: int=384, center: bool=True, window: _WindowSpec='hann') -> np.ndarray:
    """Compute the Fourier tempogram: the short-time Fourier transform of the
    onset strength envelope. [#]_

    .. [#] Grosche, Peter, Meinard Müller, and Frank Kurth.
        "Cyclic tempogram - A mid-level tempo representation for music signals."
        ICASSP, 2010.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        Audio time series.  Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    onset_envelope : np.ndarray [shape=(..., n)] or None
        Optional pre-computed onset strength envelope as provided by
        ``librosa.onset.onset_strength``.
        Multi-channel is supported.
    hop_length : int > 0
        number of audio samples between successive onset measurements
    win_length : int > 0
        length of the onset window (in frames/onset measurements)
        The default settings (384) corresponds to ``384 * hop_length / sr ~= 8.9s``.
    center : bool
        If `True`, onset windows are centered.
        If `False`, windows are left-aligned.
    window : string, function, number, tuple, or np.ndarray [shape=(win_length,)]
        A window specification as in `stft`.

    Returns
    -------
    tempogram : np.ndarray [shape=(..., win_length // 2 + 1, n)]
        Complex short-time Fourier transform of the onset envelope.

    Raises
    ------
    ParameterError
        if neither ``y`` nor ``onset_envelope`` are provided

        if ``win_length < 1``

    See Also
    --------
    tempogram
    librosa.onset.onset_strength
    librosa.util.normalize
    librosa.stft

    Examples
    --------
    >>> # Compute local onset autocorrelation
    >>> y, sr = librosa.load(librosa.ex('nutcracker'))
    >>> hop_length = 512
    >>> oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    >>> tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr,
    ...                                               hop_length=hop_length)
    >>> # Compute the auto-correlation tempogram, unnormalized to make comparison easier
    >>> ac_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
    ...                                          hop_length=hop_length, norm=None)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> ax[0].plot(librosa.times_like(oenv), oenv, label='Onset strength')
    >>> ax[0].legend(frameon=True)
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(np.abs(tempogram), sr=sr, hop_length=hop_length,
    >>>                          x_axis='time', y_axis='fourier_tempo', cmap='magma',
    ...                          ax=ax[1])
    >>> ax[1].set(title='Fourier tempogram')
    >>> ax[1].label_outer()
    >>> librosa.display.specshow(ac_tempogram, sr=sr, hop_length=hop_length,
    >>>                          x_axis='time', y_axis='tempo', cmap='magma',
    ...                          ax=ax[2])
    >>> ax[2].set(title='Autocorrelation tempogram')
    """
    from ..onset import onset_strength
    if win_length < 1:
        raise ParameterError('win_length must be a positive integer')
    if onset_envelope is None:
        if y is None:
            raise ParameterError('Either y or onset_envelope must be provided')
        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)
    return stft(onset_envelope, n_fft=win_length, hop_length=1, center=center, window=window)