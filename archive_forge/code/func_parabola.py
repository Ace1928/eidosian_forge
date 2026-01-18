import warnings
from ase.units import kJ
import numpy as np
from scipy.optimize import curve_fit
def parabola(x, a, b, c):
    """parabola polynomial function

    this function is used to fit the data to get good guesses for
    the equation of state fits

    a 4th order polynomial fit to get good guesses for
    was not a good idea because for noisy data the fit is too wiggly
    2nd order seems to be sufficient, and guarantees a single minimum"""
    return a + b * x + c * x ** 2