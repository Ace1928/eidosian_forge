from scipy._lib.deprecation import _deprecated
from scipy._lib._finite_differences import _central_diff_weights, _derivative
from numpy import array, frombuffer, load
@_deprecated(msg='scipy.misc.electrocardiogram has been deprecated in SciPy v1.10.0; and will be completely removed in SciPy v1.12.0. Dataset methods have moved into the scipy.datasets module. Use scipy.datasets.electrocardiogram instead.')
def electrocardiogram():
    """
    Load an electrocardiogram as an example for a 1-D signal.

    The returned signal is a 5 minute long electrocardiogram (ECG), a medical
    recording of the heart's electrical activity, sampled at 360 Hz.

    .. deprecated:: 1.10.0
        `electrocardiogram` has been deprecated from
        `scipy.misc.electrocardiogram` in SciPy 1.10.0 and it will be
        completely removed in SciPy 1.12.0.
        Dataset methods have moved into the `scipy.datasets` module.
        Use `scipy.datasets.electrocardiogram` instead.

    Returns
    -------
    ecg : ndarray
        The electrocardiogram in millivolt (mV) sampled at 360 Hz.

    Notes
    -----
    The provided signal is an excerpt (19:35 to 24:35) from the `record 208`_
    (lead MLII) provided by the MIT-BIH Arrhythmia Database [1]_ on
    PhysioNet [2]_. The excerpt includes noise induced artifacts, typical
    heartbeats as well as pathological changes.

    .. _record 208: https://physionet.org/physiobank/database/html/mitdbdir/records.htm#208

    .. versionadded:: 1.1.0

    References
    ----------
    .. [1] Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
           IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
           (PMID: 11446209); :doi:`10.13026/C2F305`
    .. [2] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
           Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank,
           PhysioToolkit, and PhysioNet: Components of a New Research Resource
           for Complex Physiologic Signals. Circulation 101(23):e215-e220;
           :doi:`10.1161/01.CIR.101.23.e215`

    Examples
    --------
    >>> from scipy.misc import electrocardiogram
    >>> ecg = electrocardiogram()
    >>> ecg
    array([-0.245, -0.215, -0.185, ..., -0.405, -0.395, -0.385])
    >>> ecg.shape, ecg.mean(), ecg.std()
    ((108000,), -0.16510875, 0.5992473991177294)

    As stated the signal features several areas with a different morphology.
    E.g., the first few seconds show the electrical activity of a heart in
    normal sinus rhythm as seen below.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> fs = 360
    >>> time = np.arange(ecg.size) / fs
    >>> plt.plot(time, ecg)
    >>> plt.xlabel("time in s")
    >>> plt.ylabel("ECG in mV")
    >>> plt.xlim(9, 10.2)
    >>> plt.ylim(-1, 1.5)
    >>> plt.show()

    After second 16, however, the first premature ventricular contractions, also
    called extrasystoles, appear. These have a different morphology compared to
    typical heartbeats. The difference can easily be observed in the following
    plot.

    >>> plt.plot(time, ecg)
    >>> plt.xlabel("time in s")
    >>> plt.ylabel("ECG in mV")
    >>> plt.xlim(46.5, 50)
    >>> plt.ylim(-2, 1.5)
    >>> plt.show()

    At several points large artifacts disturb the recording, e.g.:

    >>> plt.plot(time, ecg)
    >>> plt.xlabel("time in s")
    >>> plt.ylabel("ECG in mV")
    >>> plt.xlim(207, 215)
    >>> plt.ylim(-2, 3.5)
    >>> plt.show()

    Finally, examining the power spectrum reveals that most of the biosignal is
    made up of lower frequencies. At 60 Hz the noise induced by the mains
    electricity can be clearly observed.

    >>> from scipy.signal import welch
    >>> f, Pxx = welch(ecg, fs=fs, nperseg=2048, scaling="spectrum")
    >>> plt.semilogy(f, Pxx)
    >>> plt.xlabel("Frequency in Hz")
    >>> plt.ylabel("Power spectrum of the ECG in mV**2")
    >>> plt.xlim(f[[0, -1]])
    >>> plt.show()
    """
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'ecg.dat')
    with load(file_path) as file:
        ecg = file['ecg'].astype(int)
    ecg = (ecg - 1024) / 200.0
    return ecg