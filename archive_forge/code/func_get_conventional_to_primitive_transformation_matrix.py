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
@cite_conventional_cell_algo
def get_conventional_to_primitive_transformation_matrix(self, international_monoclinic=True):
    """Gives the transformation matrix to transform a conventional unit cell to a
        primitive cell according to certain standards the standards are defined in
        Setyawan, W., & Curtarolo, S. (2010). High-throughput electronic band structure
        calculations: Challenges and tools. Computational Materials Science, 49(2),
        299-312. doi:10.1016/j.commatsci.2010.05.010.

        Args:
            international_monoclinic (bool): Whether to convert to proper international convention
                such that beta is the non-right angle.

        Returns:
            Transformation matrix to go from conventional to primitive cell
        """
    conv = self.get_conventional_standard_structure(international_monoclinic=international_monoclinic)
    lattice = self.get_lattice_type()
    if 'P' in self.get_space_group_symbol() or lattice == 'hexagonal':
        return np.eye(3)
    if lattice == 'rhombohedral':
        lengths = conv.lattice.lengths
        if abs(lengths[0] - lengths[2]) < 0.0001:
            transf = np.eye
        else:
            transf = np.array([[-1, 1, 1], [2, 1, 1], [-1, -2, 1]], dtype=np.float64) / 3
    elif 'I' in self.get_space_group_symbol():
        transf = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.float64) / 2
    elif 'F' in self.get_space_group_symbol():
        transf = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64) / 2
    elif 'C' in self.get_space_group_symbol() or 'A' in self.get_space_group_symbol():
        if self.get_crystal_system() == 'monoclinic':
            transf = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]], dtype=np.float64) / 2
        else:
            transf = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]], dtype=np.float64) / 2
    else:
        transf = np.eye(3)
    return transf