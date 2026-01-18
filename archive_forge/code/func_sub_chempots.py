from __future__ import annotations
import copy
import itertools
import random
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol
from sympy.solvers import linsolve, solve
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.surface import get_slab_regions
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
def sub_chempots(gamma_dict, chempots):
    """
    Uses dot product of numpy array to sub chemical potentials
        into the surface grand potential. This is much faster
        than using the subs function in sympy.

    Args:
        gamma_dict (dict): Surface grand potential equation
            as a coefficient dictionary
        chempots (dict): Dictionary assigning each chemical
            potential (key) in gamma a value

    Returns:
        Surface energy as a float
    """
    coeffs = [gamma_dict[k] for k in gamma_dict]
    chempot_vals = []
    for k in gamma_dict:
        if k not in chempots:
            chempot_vals.append(k)
        elif k == 1:
            chempot_vals.append(1)
        else:
            chempot_vals.append(chempots[k])
    return np.dot(coeffs, chempot_vals)