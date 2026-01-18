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
def cr_class(self) -> Union[str, None]:
    """Return covalent radii class for atom or None."""
    akl = self.akl
    atmNdx = self.fields.atm
    try:
        return residue_atom_bond_state['X'][akl[atmNdx]]
    except KeyError:
        try:
            resNdx = self.fields.resname
            return residue_atom_bond_state[akl[resNdx]][akl[atmNdx]]
        except KeyError:
            return 'Hsb' if akl[atmNdx][0] == 'H' else None