from __future__ import annotations
import logging
from collections import defaultdict
import numpy as np
from monty.dev import deprecated
from scipy.constants import physical_constants
from scipy.integrate import quadrature
from scipy.misc import derivative
from scipy.optimize import minimize
from pymatgen.analysis.eos import EOS, PolynomialEOS
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.due import Doi, due
@cite_gibbs
@staticmethod
def debye_integral(y):
    """
        Debye integral. Eq(5) in  doi.org/10.1016/j.comphy.2003.12.001.

        Args:
            y (float): Debye temperature / T, upper limit

        Returns:
            float: unitless
        """
    factor = 3.0 / y ** 3
    if y < 155:
        integral = quadrature(lambda x: x ** 3 / (np.exp(x) - 1.0), 0, y)
        return next(iter(integral)) * factor
    return 6.493939 * factor