import numpy as np
import numpy.linalg as la
def dK_dl_h(self, x1, x2):
    """Returns the derivative of the hessian of the kernel function respect
        to l
        """
    I = np.identity(self.D)
    P = np.outer(x1 - x2, x1 - x2) / self.l ** 2
    prefactor = 1 - 0.5 * self.squared_distance(x1, x2)
    return -2 * (prefactor * (I - P) - P) / self.l ** 3