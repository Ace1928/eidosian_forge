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
def run_static(self, fmax):
    X = self.neb.get_positions().reshape(-1)
    for step in range(self.max_steps):
        F = self.force_function(X)
        if self.neb.get_residual() <= fmax:
            return True
        X += self.alpha * F
        self.callback(X)
    return False