from __future__ import annotations
import itertools
import math
import warnings
from typing import TYPE_CHECKING, Literal
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import factorial
from pymatgen.analysis.elasticity.strain import Strain
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.core.tensors import DEFAULT_QUAD, SquareTensor, Tensor, TensorCollection, get_uvec
from pymatgen.core.units import Unit
from pymatgen.util.due import Doi, due
def get_gruneisen_parameter(self, temperature=None, structure=None, quad=None):
    """
        Gets the single average gruneisen parameter from the TGT.

        Args:
            temperature (float): Temperature in kelvin, if not specified
                will return non-cv-normalized value
            structure (float): Structure to be used in directional heat
                capacity determination, only necessary if temperature
                is specified
            quadct (dict): quadrature for integration, should be
                dictionary with "points" and "weights" keys defaults
                to quadpy.sphere.Lebedev(19) as read from file
        """
    return np.trace(self.get_tgt(temperature, structure, quad)) / 3.0