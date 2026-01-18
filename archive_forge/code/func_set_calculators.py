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
def set_calculators(self, calculators):
    """Set new calculators to the images.

        Parameters
        ----------
        calculators : Calculator / list(Calculator)
            calculator(s) to attach to images
              - single calculator, only if allow_shared_calculator=True
            list of calculators if length:
              - length nimages, set to all images
              - length nimages-2, set to non-end images only
        """
    if not isinstance(calculators, list):
        if self.allow_shared_calculator:
            calculators = [calculators] * self.nimages
        else:
            raise RuntimeError('Cannot set shared calculator to NEB with allow_shared_calculator=False')
    n = len(calculators)
    if n == self.nimages:
        for i in range(self.nimages):
            self.images[i].calc = calculators[i]
    elif n == self.nimages - 2:
        for i in range(1, self.nimages - 1):
            self.images[i].calc = calculators[i - 1]
    else:
        raise RuntimeError('len(calculators)=%d does not fit to len(images)=%d' % (n, self.nimages))