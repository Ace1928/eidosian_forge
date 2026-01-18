import numpy as np
import numpy.linalg as la
def dK_dweight(self, X):
    """Return the derivative of K(X,X) respect to the weight """
    return self.K(X, X) * 2 / self.weight