import numpy as np
from . import kernels
def conf(self, x):
    """
        Returns the fitted curve and 1-sigma upper and lower point-wise
        confidence.
        These bounds are based on variance only, and do not include the bias.
        If the bandwidth is much larger than the curvature of the underlying
        function then the bias could be large.

        x is the points on which you want to evaluate the fit and the errors.

        Alternatively if x is specified as a positive integer, then the fit and
        confidence bands points will be returned after every
        xth sample point - so they are closer together where the data
        is denser.
        """
    if isinstance(x, int):
        sorted_x = np.array(self.x)
        sorted_x.sort()
        confx = sorted_x[::x]
        conffit = self.conf(confx)
        return (confx, conffit)
    else:
        return np.array([self.Kernel.smoothconf(self.x, self.y, xx) for xx in x])