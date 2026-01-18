import numpy as np
import numpy.linalg as la
def kernel_function_hessian(self, x1, x2):
    """Second derivatives matrix of the kernel function"""
    P = np.outer(x1 - x2, x1 - x2) / self.l ** 2
    prefactor = (np.identity(self.D) - P) / self.l ** 2
    return prefactor