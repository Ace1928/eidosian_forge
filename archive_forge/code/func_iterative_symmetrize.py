from __future__ import annotations
import copy
import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import cos, sin
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import scipy.cluster
import spglib
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule, PeriodicSite, Structure
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list, pbc_diff
from pymatgen.util.due import Doi, due
def iterative_symmetrize(mol, max_n=10, tolerance=0.3, epsilon=0.01):
    """Returns a symmetrized molecule.

    The equivalent atoms obtained via
    :meth:`~pymatgen.symmetry.analyzer.PointGroupAnalyzer.get_equivalent_atoms`
    are rotated, mirrored... unto one position.
    Then the average position is calculated.
    The average position is rotated, mirrored... back with the inverse
    of the previous symmetry operations, which gives the
    symmetrized molecule

    Args:
        mol (Molecule): A pymatgen Molecule instance.
        max_n (int): Maximum number of iterations.
        tolerance (float): Tolerance for detecting symmetry.
            Gets passed as Argument into
            ~pymatgen.analyzer.symmetry.PointGroupAnalyzer.
        epsilon (float): If the element-wise absolute difference of two
            subsequently symmetrized structures is smaller epsilon,
            the iteration stops before max_n is reached.


    Returns:
        dict: with three possible keys:
            sym_mol: A symmetrized molecule instance.
            eq_sets: A dictionary of indices mapping to sets of indices, each key maps to indices
                of all equivalent atoms. The keys are guaranteed to be not equivalent.
            sym_ops: Twofold nested dictionary. operations[i][j] gives the symmetry operation
                that maps atom i unto j.
    """
    new = mol
    n = 0
    finished = False
    eq = {'sym_mol': new, 'eq_sets': {}, 'sym_ops': {}}
    while not finished and n <= max_n:
        previous = new
        PA = PointGroupAnalyzer(previous, tolerance=tolerance)
        eq = PA.symmetrize_molecule()
        new = eq['sym_mol']
        finished = np.allclose(new.cart_coords, previous.cart_coords, atol=epsilon)
        n += 1
    return eq