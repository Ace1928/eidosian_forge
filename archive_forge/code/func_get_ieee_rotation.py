from __future__ import annotations
import collections
import itertools
import os
import string
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from monty.serialization import loadfn
from scipy.linalg import polar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def get_ieee_rotation(structure, refine_rotation=True):
    """Given a structure associated with a tensor, determines
        the rotation matrix for IEEE conversion according to
        the 1987 IEEE standards.

        Args:
            structure (Structure): a structure associated with the
                tensor to be converted to the IEEE standard
            refine_rotation (bool): whether to refine the rotation
                using SquareTensor.refine_rotation
        """
    sga = SpacegroupAnalyzer(structure)
    dataset = sga.get_symmetry_dataset()
    trans_mat = dataset['transformation_matrix']
    conv_latt = Lattice(np.transpose(np.dot(np.transpose(structure.lattice.matrix), np.linalg.inv(trans_mat))))
    xtal_sys = sga.get_crystal_system()
    vecs = conv_latt.matrix
    lengths = np.array(conv_latt.abc)
    angles = np.array(conv_latt.angles)
    rotation = np.zeros((3, 3))
    if xtal_sys == 'cubic':
        rotation = [vecs[i] / lengths[i] for i in range(3)]
    elif xtal_sys == 'tetragonal':
        rotation = np.array([vec / mag for mag, vec in sorted(zip(lengths, vecs), key=lambda x: x[0])])
        if abs(lengths[2] - lengths[1]) < abs(lengths[1] - lengths[0]):
            rotation[0], rotation[2] = (rotation[2], rotation[0].copy())
        rotation[1] = get_uvec(np.cross(rotation[2], rotation[0]))
    elif xtal_sys == 'orthorhombic':
        rotation = [vec / mag for mag, vec in sorted(zip(lengths, vecs))]
        rotation = np.roll(rotation, 2, axis=0)
    elif xtal_sys in ('trigonal', 'hexagonal'):
        tf_index = np.argmin(abs(angles - 120.0))
        non_tf_mask = np.logical_not(angles == angles[tf_index])
        rotation[2] = get_uvec(vecs[tf_index])
        rotation[0] = get_uvec(vecs[non_tf_mask][0])
        rotation[1] = get_uvec(np.cross(rotation[2], rotation[0]))
    elif xtal_sys == 'monoclinic':
        u_index = np.argmax(abs(angles - 90.0))
        n_umask = np.logical_not(angles == angles[u_index])
        rotation[1] = get_uvec(vecs[u_index])
        c = next((vec / mag for mag, vec in sorted(zip(lengths[n_umask], vecs[n_umask]))))
        rotation[2] = np.array(c)
        rotation[0] = np.cross(rotation[1], rotation[2])
    elif xtal_sys == 'triclinic':
        rotation = [vec / mag for mag, vec in sorted(zip(lengths, vecs))]
        rotation[1] = get_uvec(np.cross(rotation[2], rotation[0]))
        rotation[0] = np.cross(rotation[1], rotation[2])
    rotation = SquareTensor(rotation)
    if refine_rotation:
        rotation = rotation.refine_rotation()
    return rotation