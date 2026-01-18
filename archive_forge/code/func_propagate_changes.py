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
def propagate_changes(self) -> None:
    """Track through di/hedra to invalidate dependent atoms."""
    csNdx = 0
    csLen = len(self.initNCaCs)
    atmNdx = AtomKey.fields.atm
    posNdx = AtomKey.fields.respos
    done = set()
    while csNdx < csLen:
        startAK = self.initNCaCs[csNdx][0]
        csStart = self.atomArrayIndex[startAK]
        csnTry = csNdx + 1
        if csLen == csnTry:
            csNext = self.AAsiz
        else:
            finAK = self.initNCaCs[csnTry][0]
            csNext = self.atomArrayIndex[finAK]
        for andx in range(csStart, csNext):
            if not self.atomArrayValid[andx]:
                ak = self.aktuple[andx]
                atm = ak.akl[atmNdx]
                pos = ak.akl[posNdx]
                if atm in ('N', 'CA', 'C'):
                    self.atomArrayValid[andx:csNext] = False
                    break
                elif pos not in done and atm != 'H':
                    for i in range(andx, csNext):
                        if self.aktuple[i].akl[posNdx] == pos:
                            self.atomArrayValid[i] = False
                        else:
                            break
                    done.add(pos)
        csNdx += 1