import re
from collections import deque, namedtuple
import copy
from numbers import Integral
import numpy as np  # type: ignore
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.vectors import multi_coord_space, multi_rot_Z
from Bio.PDB.vectors import coord_space
from Bio.PDB.ic_data import ic_data_backbone, ic_data_sidechains
from Bio.PDB.ic_data import primary_angles
from Bio.PDB.ic_data import ic_data_sidechain_extras, residue_atom_bond_state
from Bio.PDB.ic_data import dihedra_primary_defaults, hedra_defaults
from typing import (
@staticmethod
def angle_avg(alst: List, in_rads: bool=False, out_rads: bool=False):
    """Get average of list of +/-180 angles.

        :param List alst: list of angles to average
        :param bool in_rads: input values are in radians
        :param bool out_rads: report result in radians
        """
    walst = alst if in_rads else np.deg2rad(alst)
    ravg = np.arctan2(np.sum(np.sin(walst)), np.sum(np.cos(walst)))
    return ravg if out_rads else np.rad2deg(ravg)