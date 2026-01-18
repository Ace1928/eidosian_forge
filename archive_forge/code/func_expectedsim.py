import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
def expectedsim(self, func, nobs=100, T=1, dt=None, nrepl=1):
    """get expectation of a function of a Wiener Process by simulation

        initially test example from
        """
    W, t = self.simulateW(nobs=nobs, T=T, dt=dt, nrepl=nrepl)
    U = func(t, W)
    Umean = U.mean(0)
    return (U, Umean, t)