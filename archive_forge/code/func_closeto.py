import math
import numpy as np
def closeto(self, ms, edge):
    if self._offset > 0:
        digits = np.log10(self._offset / self.step)
        tol = max(1e-10, 10 ** (digits - 12))
        tol = min(0.4999, tol)
    else:
        tol = 1e-10
    return abs(ms - edge) < tol