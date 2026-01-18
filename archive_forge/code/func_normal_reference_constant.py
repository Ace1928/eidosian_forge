from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
@property
def normal_reference_constant(self):
    """
        Constant used for silverman normal reference asymtotic bandwidth
        calculation.

        C  = 2((pi^(1/2)*(nu!)^3 R(k))/(2nu(2nu)!kap_nu(k)^2))^(1/(2nu+1))
        nu = kernel order
        kap_nu = nu'th moment of kernel
        R = kernel roughness (square of L^2 norm)

        Note: L2Norm property returns square of norm.
        """
    nu = self._order
    if not nu == 2:
        msg = 'Only implemented for second order kernels'
        raise NotImplementedError(msg)
    if self._normal_reference_constant is None:
        C = np.pi ** 0.5 * factorial(nu) ** 3 * self.L2Norm
        C /= 2 * nu * factorial(2 * nu) * self.moments(nu) ** 2
        C = 2 * C ** (1.0 / (2 * nu + 1))
        self._normal_reference_constant = C
    return self._normal_reference_constant