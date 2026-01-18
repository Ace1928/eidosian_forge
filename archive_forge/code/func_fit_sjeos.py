import warnings
from ase.units import kJ
import numpy as np
from scipy.optimize import curve_fit
def fit_sjeos(self):
    """Calculate volume, energy, and bulk modulus.

        Returns the optimal volume, the minimum energy, and the bulk
        modulus.  Notice that the ASE units for the bulk modulus is
        eV/Angstrom^3 - to get the value in GPa, do this::

          v0, e0, B = eos.fit()
          print(B / kJ * 1.0e24, 'GPa')

        """
    fit0 = np.poly1d(np.polyfit(self.v ** (-(1 / 3)), self.e, 3))
    fit1 = np.polyder(fit0, 1)
    fit2 = np.polyder(fit1, 1)
    self.v0 = None
    for t in np.roots(fit1):
        if isinstance(t, float) and t > 0 and (fit2(t) > 0):
            self.v0 = t ** (-3)
            break
    if self.v0 is None:
        raise ValueError('No minimum!')
    self.e0 = fit0(t)
    self.B = t ** 5 * fit2(t) / 9
    self.fit0 = fit0
    return (self.v0, self.e0, self.B)