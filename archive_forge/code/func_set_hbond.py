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
def set_hbond(self) -> None:
    """For OpenSCAD, mark H-N and C-O bonds to be hbonds (magnets).

        See :func:`.SCADIO.write_SCAD`
        """
    for h in self.hedra.values():
        if h.e_class == 'HNCA':
            h.hbond_1 = True
        elif h.e_class == 'CACO':
            h.hbond_2 = True