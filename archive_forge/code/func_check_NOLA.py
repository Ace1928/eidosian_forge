import warnings
import cupy
from cupyx.scipy.signal.windows._windows import get_window
from cupyx.scipy.signal._spectral_impl import (
def check_NOLA(window, nperseg, noverlap, tol=1e-10):
    """Check whether the Nonzero Overlap Add (NOLA) constraint is met.

    Parameters
    ----------
    window : str or tuple or array_like
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg.
    nperseg : int
        Length of each segment.
    noverlap : int
        Number of points to overlap between segments.
    tol : float, optional
        The allowed variance of a bin's weighted sum from the median bin
        sum.

    Returns
    -------
    verdict : bool
        `True` if chosen combination satisfies the NOLA constraint within
        `tol`, `False` otherwise

    See Also
    --------
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint is met
    stft: Short Time Fourier Transform
    istft: Inverse Short Time Fourier Transform

    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT in
    `istft`, the signal windowing must obey the constraint of "nonzero
    overlap add" (NOLA):

    .. math:: \\sum_{t}w^{2}[n-tH] \\ne 0

    for all :math:`n`, where :math:`w` is the window function, :math:`t` is the
    frame index, and :math:`H` is the hop size (:math:`H` = `nperseg` -
    `noverlap`).

    This ensures that the normalization factors in the denominator of the
    overlap-add inversion equation are not zero. Only very pathological windows
    will fail the NOLA constraint.

    See [1]_, [2]_ for more information.

    References
    ----------
    .. [1] Julius O. Smith III, "Spectral Audio Signal Processing", W3K
           Publishing, 2011,ISBN 978-0-9745607-3-1.
    .. [2] G. Heinzel, A. Ruediger and R. Schilling, "Spectrum and
           spectral density estimation by the Discrete Fourier transform
           (DFT), including a comprehensive list of window functions and
           some new at-top windows", 2002,
           http://hdl.handle.net/11858/00-001M-0000-0013-557A-5

    """
    nperseg = int(nperseg)
    if nperseg < 1:
        raise ValueError('nperseg must be a positive integer')
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg')
    if noverlap < 0:
        raise ValueError('noverlap must be a nonnegative integer')
    noverlap = int(noverlap)
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = cupy.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of nperseg')
    step = nperseg - noverlap
    binsums = sum((win[ii * step:(ii + 1) * step] ** 2 for ii in range(nperseg // step)))
    if nperseg % step != 0:
        binsums[:nperseg % step] += win[-(nperseg % step):] ** 2
    return cupy.min(binsums) > tol