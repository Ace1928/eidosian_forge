import numpy as np
def silverman_transform(bw, M, RANGE):
    """
    FFT of Gaussian kernel following to Silverman AS 176.

    Notes
    -----
    Underflow is intentional as a dampener.
    """
    J = np.arange(M / 2 + 1)
    FAC1 = 2 * (np.pi * bw / RANGE) ** 2
    JFAC = J ** 2 * FAC1
    BC = 1 - 1.0 / 3 * (J * 1.0 / M * np.pi) ** 2
    FAC = np.exp(-JFAC) / BC
    kern_est = np.r_[FAC, FAC[1:-1]]
    return kern_est