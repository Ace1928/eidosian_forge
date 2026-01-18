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
def pdb_residue_string(self) -> str:
    """Generate PDB ATOM records for this residue as string.

        Convenience method for functionality not exposed in PDBIO.py.
        Increments :data:`IC_Residue.atom_sernum` if not None

        :param IC_Residue.atom_sernum: Class variable default None.
            Override and increment atom serial number if not None
        :param IC_Residue.atom_chain: Class variable.
            Override atom chain id if not None

        .. todo::
            move to PDBIO
        """
    str = ''
    atomArrayIndex = self.cic.atomArrayIndex
    bpAtomArray = self.cic.bpAtomArray
    respos = self.rbase[0]
    resposNdx = AtomKey.fields.respos
    for ak in sorted(self.ak_set):
        if int(ak.akl[resposNdx]) == respos:
            str += IC_Residue._pdb_atom_string(bpAtomArray[atomArrayIndex[ak]])
            if IC_Residue.atom_sernum is not None:
                IC_Residue.atom_sernum += 1
    return str