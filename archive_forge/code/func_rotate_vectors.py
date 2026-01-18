import sys
import time
import warnings
from math import cos, sin, atan, tan, degrees, pi, sqrt
from typing import Dict, Any
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.parallel import world
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import IOContext
def rotate_vectors(v1i, v2i, angle):
    """Rotate vectors *v1i* and *v2i* by *angle*"""
    cAng = cos(angle)
    sAng = sin(angle)
    v1o = v1i * cAng + v2i * sAng
    v2o = v2i * cAng - v1i * sAng
    return (normalize(v1o) * norm(v1i), normalize(v2o) * norm(v2i))