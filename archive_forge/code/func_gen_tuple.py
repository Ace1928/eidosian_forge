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
def gen_tuple(akstr: str) -> Tuple:
    """Generate AtomKey tuple for ':'-joined AtomKey string.

        Generate (2_A_C, 3_P_N, 3_P_CA) from '2_A_C:3_P_N:3_P_CA'
        :param str akstr: string of ':'-separated AtomKey strings
        """
    return tuple([AtomKey(i) for i in akstr.split(':')])