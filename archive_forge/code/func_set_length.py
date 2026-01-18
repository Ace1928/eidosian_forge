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
def set_length(self, ak_tpl: BKT, newLength: float):
    """Set bond length for specified atom pair; sets needs_update.

        :param tuple .ak_tpl: tuple of AtomKeys
            Pair of atoms in this Hedron
        """
    if 2 > len(ak_tpl):
        raise TypeError(f'Require exactly 2 AtomKeys: {ak_tpl!s}')
    elif all((ak in self.atomkeys[:2] for ak in ak_tpl)):
        self.cic.hedraL12[self.ndx] = newLength
    elif all((ak in self.atomkeys[1:] for ak in ak_tpl)):
        self.cic.hedraL23[self.ndx] = newLength
    else:
        raise TypeError('%s not found in %s' % (str(ak_tpl), self))
    self._invalidate_atoms()