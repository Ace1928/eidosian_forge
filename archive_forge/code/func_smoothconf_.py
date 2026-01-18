from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
def smoothconf_(self, xs, ys, x):
    """Returns the kernel smoothing estimate with confidence 1sigma bounds
        """
    xs, ys = self.in_domain(xs, ys, x)
    if len(xs) > 0:
        fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
        rs = square(subtract(ys, fittedvals))
        w = np.sum(square(subtract(1.0, square(divide(subtract(xs, x), self.h)))))
        v = np.sum(multiply(rs, square(subtract(1, square(divide(subtract(xs, x), self.h))))))
        var = v / w
        sd = np.sqrt(var)
        K = self.L2Norm
        yhat = self.smooth(xs, ys, x)
        err = sd * K / np.sqrt(0.9375 * w * self.h)
        return (yhat - err, yhat, yhat + err)
    else:
        return (np.nan, np.nan, np.nan)