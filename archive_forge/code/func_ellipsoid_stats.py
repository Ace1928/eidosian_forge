import numpy as np
from scipy.special import ellipkinc as ellip_F, ellipeinc as ellip_E
def ellipsoid_stats(a, b, c):
    """
    Calculates analytical surface area and volume for ellipsoid with
    semimajor axes aligned with grid dimensions of specified `spacing`.

    Parameters
    ----------
    a : float
        Length of semimajor axis aligned with x-axis.
    b : float
        Length of semimajor axis aligned with y-axis.
    c : float
        Length of semimajor axis aligned with z-axis.

    Returns
    -------
    vol : float
        Calculated volume of ellipsoid.
    surf : float
        Calculated surface area of ellipsoid.

    """
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError('Parameters a, b, and c must all be > 0')
    abc = [a, b, c]
    abc.sort(reverse=True)
    a = abc[0]
    b = abc[1]
    c = abc[2]
    vol = 4 / 3.0 * np.pi * a * b * c
    phi = np.arcsin((1.0 - c ** 2 / a ** 2.0) ** 0.5)
    d = float((a ** 2 - c ** 2) ** 0.5)
    m = a ** 2 * (b ** 2 - c ** 2) / float(b ** 2 * (a ** 2 - c ** 2))
    F = ellip_F(phi, m)
    E = ellip_E(phi, m)
    surf = 2 * np.pi * (c ** 2 + b * c ** 2 / d * F + b * d * E)
    return (vol, surf)