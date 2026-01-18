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
def is_valid_op(self, symmop) -> bool:
    """Check if a particular symmetry operation is a valid symmetry operation for a
        molecule, i.e., the operation maps all atoms to another equivalent atom.

        Args:
            symmop (SymmOp): Symmetry operation to test.

        Returns:
            bool: Whether SymmOp is valid for Molecule.
        """
    coords = self.centered_mol.cart_coords
    for site in self.centered_mol:
        coord = symmop.operate(site.coords)
        ind = find_in_coord_list(coords, coord, self.tol)
        if not (len(ind) == 1 and self.centered_mol[ind[0]].species == site.species):
            return False
    return True