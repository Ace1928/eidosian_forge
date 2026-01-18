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
@property
def compliance_tensor(self):
    """
        Returns the Voigt notation compliance tensor, which is the matrix
        inverse of the Voigt notation elastic tensor.
        """
    s_voigt = np.linalg.inv(self.voigt)
    return ComplianceTensor.from_voigt(s_voigt)