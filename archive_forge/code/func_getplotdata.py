import warnings
from ase.units import kJ
import numpy as np
from scipy.optimize import curve_fit
def getplotdata(self):
    if self.v0 is None:
        self.fit()
    x = np.linspace(min(self.v), max(self.v), 100)
    if self.eos_string == 'sj':
        y = self.fit0(x ** (-(1 / 3)))
    else:
        y = self.func(x, *self.eos_parameters)
    return (self.eos_string, self.e0, self.v0, self.B, x, y, self.v, self.e)