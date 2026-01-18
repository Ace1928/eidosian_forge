import numpy as np
import numpy.linalg as la
def kernel_function_gradient(self, x1, x2):
    """Gradient of kernel_function respect to the second entry.
        x1: first data point
        x2: second data point
        """
    prefactor = (x1 - x2) / self.l ** 2
    return prefactor