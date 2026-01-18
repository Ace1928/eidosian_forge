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
def get_neb_method(neb, method):
    if method == 'eb':
        return FullSpringMethod(neb)
    elif method == 'aseneb':
        return ASENEBMethod(neb)
    elif method == 'improvedtangent':
        return ImprovedTangentMethod(neb)
    elif method == 'spline':
        return SplineMethod(neb)
    elif method == 'string':
        return StringMethod(neb)
    else:
        raise ValueError(f'Bad method: {method}')