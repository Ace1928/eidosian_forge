import numpy as np
def c_sja(n, p):
    if p > 1 or p < -1:
        jc = np.full(3, np.nan)
    elif n > 12 or n < 1:
        jc = np.full(3, np.nan)
    elif p == -1:
        jc = ejcp0[n - 1, :]
    elif p == 0:
        jc = ejcp1[n - 1, :]
    elif p == 1:
        jc = ejcp2[n - 1, :]
    return jc