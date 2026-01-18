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
def symmetrize_molecule(self):
    """Returns a symmetrized molecule.

        The equivalent atoms obtained via
        :meth:`~pymatgen.symmetry.analyzer.PointGroupAnalyzer.get_equivalent_atoms`
        are rotated, mirrored... unto one position.
        Then the average position is calculated.
        The average position is rotated, mirrored... back with the inverse
        of the previous symmetry operations, which gives the
        symmetrized molecule

        Returns:
            dict: with three possible keys:
                sym_mol: A symmetrized molecule instance.
                eq_sets: A dictionary of indices mapping to sets of indices, each key maps to indices
                    of all equivalent atoms. The keys are guaranteed to be not equivalent.
                sym_ops: Twofold nested dictionary. operations[i][j] gives the symmetry operation
                    that maps atom i unto j.
        """
    eq = self.get_equivalent_atoms()
    eq_sets, ops = (eq['eq_sets'], eq['sym_ops'])
    coords = self.centered_mol.cart_coords.copy()
    for i, eq_indices in eq_sets.items():
        for j in eq_indices:
            coords[j] = np.dot(ops[j][i], coords[j])
        coords[i] = np.mean(coords[list(eq_indices)], axis=0)
        for j in eq_indices:
            if j == i:
                continue
            coords[j] = np.dot(ops[i][j], coords[i])
            coords[j] = np.dot(ops[i][j], coords[i])
    molecule = Molecule(species=self.centered_mol.species_and_occu, coords=coords)
    return {'sym_mol': molecule, 'eq_sets': eq_sets, 'sym_ops': ops}