import sys
import threading
import warnings
from abc import ABC, abstractmethod
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz
import ase.parallel
from ase.build import minimize_rotation_and_translation
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import MDMin
from ase.optimize.optimize import Optimizer
from ase.optimize.sciopt import OptimizerConvergenceError
from ase.geometry import find_mic
from ase.utils import lazyproperty, deprecated
from ase.utils.forcecurve import fit_images
from ase.optimize.precon import Precon, PreconImages
from ase.optimize.ode import ode12r
def integrate_forces(self, spline_points=1000, bc_type='not-a-knot'):
    """Use spline fit to integrate forces along MEP to approximate
        energy differences using the virtual work approach.

        Args:
            spline_points (int, optional): Number of points. Defaults to 1000.
            bc_type (str, optional): Boundary conditions, default 'not-a-knot'.

        Returns:
            s: reaction coordinate in range [0, 1], with `spline_points` entries
            E: result of integrating forces, on the same grid as `s`.
            F: projected forces along MEP
        """
    fit = self.spline_fit(norm='euclidean')
    forces = np.array([image.get_forces().reshape(-1) for image in self.images])
    f = CubicSpline(fit.s, forces, bc_type=bc_type)
    s = np.linspace(0.0, 1.0, spline_points, endpoint=True)
    dE = f(s) * fit.dx_ds(s)
    F = dE.sum(axis=1)
    E = -cumtrapz(F, s, initial=0.0)
    return (s, E, F)