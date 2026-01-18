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
def gen_key(lst: List['AtomKey']) -> str:
    """Generate string of ':'-joined AtomKey strings from input.

        Generate '2_A_C:3_P_N:3_P_CA' from (2_A_C, 3_P_N, 3_P_CA)
        :param list lst: list of AtomKey objects
        """
    if 4 == len(lst):
        return f'{lst[0].id}:{lst[1].id}:{lst[2].id}:{lst[3].id}'
    else:
        return f'{lst[0].id}:{lst[1].id}:{lst[2].id}'