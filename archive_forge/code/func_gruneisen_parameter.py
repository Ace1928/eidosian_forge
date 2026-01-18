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
def gruneisen_parameter(self, temperature, volume):
    """
        Slater-gamma formulation(the default):
            gruneisen parameter = - d log(theta)/ d log(V) = - (1/6 + 0.5 d log(B)/ d log(V))
                                = - (1/6 + 0.5 V/B dB/dV), where dB/dV = d^2E/dV^2 + V * d^3E/dV^3.

        Mie-gruneisen formulation:
            Eq(31) in doi.org/10.1016/j.comphy.2003.12.001
            Eq(7) in Blanco et. al. Journal of Molecular Structure (Theochem)
                368 (1996) 245-255
            Also see J.P. Poirier, Introduction to the Physics of the Earth's
                Interior, 2nd ed. (Cambridge University Press, Cambridge,
                2000) Eq(3.53)

        Args:
            temperature (float): temperature in K
            volume (float): in Ang^3

        Returns:
            float: unitless
        """
    if isinstance(self.eos, PolynomialEOS):
        p = np.poly1d(self.eos.eos_params)
        dEdV = np.polyder(p, 1)(volume)
        d2EdV2 = np.polyder(p, 2)(volume)
        d3EdV3 = np.polyder(p, 3)(volume)
    else:
        func = self.ev_eos_fit.func
        dEdV = derivative(func, volume, dx=0.001)
        d2EdV2 = derivative(func, volume, dx=0.001, n=2, order=5)
        d3EdV3 = derivative(func, volume, dx=0.001, n=3, order=7)
    if self.use_mie_gruneisen:
        p0 = dEdV
        return self.gpa_to_ev_ang * volume * (self.pressure + p0 / self.gpa_to_ev_ang) / self.vibrational_internal_energy(temperature, volume)
    dBdV = d2EdV2 + d3EdV3 * volume
    return -(1.0 / 6.0 + 0.5 * volume * dBdV / FloatWithUnit(self.ev_eos_fit.b0_GPa, 'GPa').to('eV ang^-3'))