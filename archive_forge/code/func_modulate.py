from collections import Counter
import numpy as np
from scipy import sparse
from pygsp import utils
from . import fourier, difference  # prevent circular import in Python < 3.5
def modulate(self, f, k):
    """Modulate the signal *f* to the frequency *k*.

        Parameters
        ----------
        f : ndarray
            Signal (column)
        k : int
            Index of frequencies

        Returns
        -------
        fm : ndarray
            Modulated signal

        """
    nt = np.shape(f)[1]
    fm = np.kron(np.ones((1, nt)), self.U[:, k])
    fm *= np.kron(np.ones((nt, 1)), f)
    fm *= np.sqrt(self.N)
    return fm