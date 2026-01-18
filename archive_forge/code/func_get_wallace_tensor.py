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
def get_wallace_tensor(self, tau):
    """
        Gets the Wallace Tensor for determining yield strength
        criteria.

        Args:
            tau (3x3 array-like): stress at which to evaluate
                the wallace tensor
        """
    b = 0.5 * (np.einsum('ml,kn->klmn', tau, np.eye(3)) + np.einsum('km,ln->klmn', tau, np.eye(3)) + np.einsum('nl,km->klmn', tau, np.eye(3)) + np.einsum('kn,lm->klmn', tau, np.eye(3)) + -2 * np.einsum('kl,mn->klmn', tau, np.eye(3)))
    strain = self.get_strain_from_stress(tau)
    b += self.get_effective_ecs(strain)
    return b