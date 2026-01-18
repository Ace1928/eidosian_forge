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
@classmethod
def from_independent_strains(cls, strains, stresses, eq_stress=None, vasp=False, tol: float=1e-10) -> Self:
    """
        Constructs the elastic tensor least-squares fit of independent strains

        Args:
            strains (list of Strains): list of strain objects to fit
            stresses (list of Stresses): list of stress objects to use in fit
                corresponding to the list of strains
            eq_stress (Stress): equilibrium stress to use in fitting
            vasp (bool): flag for whether the stress tensor should be
                converted based on vasp units/convention for stress
            tol (float): tolerance for removing near-zero elements of the
                resulting tensor.
        """
    strain_states = [tuple(ss) for ss in np.eye(6)]
    ss_dict = get_strain_state_dict(strains, stresses, eq_stress=eq_stress)
    if not set(strain_states) <= set(ss_dict):
        raise ValueError(f'Missing independent strain states: {set(strain_states) - set(ss_dict)}')
    if len(set(ss_dict) - set(strain_states)) > 0:
        warnings.warn('Extra strain states in strain-stress pairs are neglected in independent strain fitting')
    c_ij = np.zeros((6, 6))
    for ii in range(6):
        strains = ss_dict[strain_states[ii]]['strains']
        stresses = ss_dict[strain_states[ii]]['stresses']
        for jj in range(6):
            c_ij[ii, jj] = np.polyfit(strains[:, ii], stresses[:, jj], 1)[0]
    if vasp:
        c_ij *= -0.1
    instance = cls.from_voigt(c_ij)
    return instance.zeroed(tol)