from __future__ import annotations
import itertools
import os
from typing import TYPE_CHECKING
import numpy as np
from matplotlib import patches
from matplotlib.path import Path
from monty.serialization import loadfn
from scipy.spatial import Delaunay
from pymatgen import vis
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Molecule, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.core.surface import generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list_pbc
def get_rot(slab: Slab) -> SymmOp:
    """Gets the transformation to rotate the z axis into the miller index."""
    new_z = get_mi_vec(slab)
    a, _b, _c = slab.lattice.matrix
    new_x = a / np.linalg.norm(a)
    new_y = np.cross(new_z, new_x)
    x, y, z = np.eye(3)
    rot_matrix = np.array([np.dot(*el) for el in itertools.product([x, y, z], [new_x, new_y, new_z])]).reshape(3, 3)
    rot_matrix = np.transpose(rot_matrix)
    return SymmOp.from_rotation_and_translation(rot_matrix)