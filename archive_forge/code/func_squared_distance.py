import numpy as np
import numpy.linalg as la
def squared_distance(self, x1, x2):
    """Returns the norm of x1-x2 using diag(l) as metric """
    return np.sum((x1 - x2) * (x1 - x2)) / self.l ** 2