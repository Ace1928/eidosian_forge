import numpy as np
import numpy.linalg as la
def dK_dl(self, X):
    """Return the derivative of K(X,X) respect of l"""
    return np.block([[self.dK_dl_matrix(x1, x2) for x2 in X] for x1 in X])