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
def applyMtx(self, mtx: np.array) -> None:
    """Apply matrix to atom_coords for this IC_Residue."""
    aa = self.cic.atomArray
    aai = self.cic.atomArrayIndex
    rpndx = AtomKey.fields.respos
    rp = str(self.rbase[0])
    aselect = [aai.get(k) for k in aai.keys() if k.akl[rpndx] == rp]
    aas = aa[aselect]
    aa[aselect] = aas.dot(mtx.transpose())
    '\n        # slower way, one at a time\n        for ak in sorted(self.ak_set):\n            ndx = self.cic.atomArrayIndex[ak]\n            self.cic.atomArray[ndx] = mtx.dot(self.cic.atomArray[ndx])\n        '