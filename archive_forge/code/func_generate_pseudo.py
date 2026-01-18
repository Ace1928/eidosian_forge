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
def generate_pseudo(strain_states, order=3):
    """Generates the pseudo-inverse for a given set of strains.

    Args:
        strain_states (6xN array like): a list of Voigt-notation strain-states,
            i. e. perturbed indices of the strain as a function of the smallest
            strain e. g. (0, 1, 0, 0, 1, 0)
        order (int): order of pseudo-inverse to calculate

    Returns:
        pseudo_inverses: for each order tensor, these can be multiplied by the central
            difference derivative of the stress with respect to the strain state
        absent_syms: symbols of the tensor absent from the PI expression
    """
    symb = sp.Symbol('s')
    n_states = len(strain_states)
    n_i = np.array(strain_states) * symb
    pseudo_inverses, absent_symbols = ([], [])
    for degree in range(2, order + 1):
        c_vec, c_arr = get_symbol_list(degree)
        s_arr = np.zeros((n_states, 6), dtype=object)
        for n, strain_v in enumerate(n_i):
            exps = c_arr.copy()
            for _ in range(degree - 1):
                exps = np.dot(exps, strain_v)
            exps /= math.factorial(degree - 1)
            s_arr[n] = [sp.diff(exp, symb, degree - 1) for exp in exps]
        s_vec = s_arr.ravel()
        present_symbols = set.union(*(exp.atoms(sp.Symbol) for exp in s_vec))
        absent_symbols += [set(c_vec) - present_symbols]
        pseudo_mat = np.zeros((6 * n_states, len(c_vec)))
        for n, c in enumerate(c_vec):
            pseudo_mat[:, n] = v_diff(s_vec, c)
        pseudo_inverses.append(np.linalg.pinv(pseudo_mat))
    return (pseudo_inverses, absent_symbols)