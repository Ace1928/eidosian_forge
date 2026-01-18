import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat
def getisinvertible(self, a=None):
    """check whether the auto-regressive lag-polynomial is stationary

        Returns
        -------
        isinvertible : bool

        *attaches*

        maeigenvalues : complex array
            eigenvalues sorted by absolute value

        References
        ----------
        formula taken from NAG manual

        """
    if a is not None:
        a = a
    elif self.isindependent:
        a = self.reduceform(self.ma)[1:]
    else:
        a = self.ma[1:]
    if a.shape[0] == 0:
        self.maeigenvalues = np.array([], np.complex)
        return True
    amat = self.stacksquare(a)
    ev = np.sort(np.linalg.eigvals(amat))[::-1]
    self.maeigenvalues = ev
    return (np.abs(ev) < 1).all()