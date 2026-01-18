import numpy as np
from .._shared.utils import _supported_float_type
from .colorconv import lab2lch, _cart2polar_2pi
def deltaE_ciede2000(lab1, lab2, kL=1, kC=1, kH=1, *, channel_axis=-1):
    """Color difference as given by the CIEDE 2000 standard.

    CIEDE 2000 is a major revision of CIDE94.  The perceptual calibration is
    largely based on experience with automotive paint on smooth surfaces.

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    kL : float (range), optional
        lightness scale factor, 1 for "acceptably close"; 2 for "imperceptible"
        see deltaE_cmc
    kC : float (range), optional
        chroma scale factor, usually 1
    kH : float (range), optional
        hue scale factor, usually 1
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    deltaE : array_like
        The distance between `lab1` and `lab2`

    Notes
    -----
    CIEDE 2000 assumes parametric weighting factors for the lightness, chroma,
    and hue (`kL`, `kC`, `kH` respectively).  These default to 1.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf
           :DOI:`10.1364/AO.33.008069`
    .. [3] M. Melgosa, J. Quesada, and E. Hita, "Uniformity of some recent
           color metrics tested with an accurate color-difference tolerance
           dataset," Appl. Opt. 33, 8069-8077 (1994).
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    channel_axis = channel_axis % lab1.ndim
    unroll = False
    if lab1.ndim == 1 and lab2.ndim == 1:
        unroll = True
        if lab1.ndim == 1:
            lab1 = lab1[None, :]
        if lab2.ndim == 1:
            lab2 = lab2[None, :]
        channel_axis += 1
    L1, a1, b1 = np.moveaxis(lab1, source=channel_axis, destination=0)[:3]
    L2, a2, b2 = np.moveaxis(lab2, source=channel_axis, destination=0)[:3]
    Cbar = 0.5 * (np.hypot(a1, b1) + np.hypot(a2, b2))
    c7 = Cbar ** 7
    G = 0.5 * (1 - np.sqrt(c7 / (c7 + 25 ** 7)))
    scale = 1 + G
    C1, h1 = _cart2polar_2pi(a1 * scale, b1)
    C2, h2 = _cart2polar_2pi(a2 * scale, b2)
    Lbar = 0.5 * (L1 + L2)
    tmp = (Lbar - 50) ** 2
    SL = 1 + 0.015 * tmp / np.sqrt(20 + tmp)
    L_term = (L2 - L1) / (kL * SL)
    Cbar = 0.5 * (C1 + C2)
    SC = 1 + 0.045 * Cbar
    C_term = (C2 - C1) / (kC * SC)
    h_diff = h2 - h1
    h_sum = h1 + h2
    CC = C1 * C2
    dH = h_diff.copy()
    dH[h_diff > np.pi] -= 2 * np.pi
    dH[h_diff < -np.pi] += 2 * np.pi
    dH[CC == 0.0] = 0.0
    dH_term = 2 * np.sqrt(CC) * np.sin(dH / 2)
    Hbar = h_sum.copy()
    mask = np.logical_and(CC != 0.0, np.abs(h_diff) > np.pi)
    Hbar[mask * (h_sum < 2 * np.pi)] += 2 * np.pi
    Hbar[mask * (h_sum >= 2 * np.pi)] -= 2 * np.pi
    Hbar[CC == 0.0] *= 2
    Hbar *= 0.5
    T = 1 - 0.17 * np.cos(Hbar - np.deg2rad(30)) + 0.24 * np.cos(2 * Hbar) + 0.32 * np.cos(3 * Hbar + np.deg2rad(6)) - 0.2 * np.cos(4 * Hbar - np.deg2rad(63))
    SH = 1 + 0.015 * Cbar * T
    H_term = dH_term / (kH * SH)
    c7 = Cbar ** 7
    Rc = 2 * np.sqrt(c7 / (c7 + 25 ** 7))
    dtheta = np.deg2rad(30) * np.exp(-((np.rad2deg(Hbar) - 275) / 25) ** 2)
    R_term = -np.sin(2 * dtheta) * Rc * C_term * H_term
    dE2 = L_term ** 2
    dE2 += C_term ** 2
    dE2 += H_term ** 2
    dE2 += R_term
    ans = np.sqrt(np.maximum(dE2, 0))
    if unroll:
        ans = ans[0]
    return ans