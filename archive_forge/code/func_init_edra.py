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
def init_edra(self, verbose: bool=False) -> None:
    """Create chain and residue di/hedra structures, arrays, atomArray.

        Inputs:
            self.ordered_aa_ic_list : list of IC_Residue
        Generates:
            * edra objects, self.di/hedra (executes :meth:`._create_edra`)
            * atomArray and support (executes :meth:`.build_atomArray`)
            * self.hedraLen : number of hedra in structure
            * hedraL12 : numpy arrays for lengths, angles (empty)
            * hedraAngle ..
            * hedraL23 ..
            * self.hedraNdx : dict mapping hedrakeys to hedraL12 etc
            * self.dihedraLen : number of dihedra in structure
            * dihedraAngle ..
            * dihedraAngleRads : np arrays for angles (empty)
            * self.dihedraNdx : dict mapping dihedrakeys to dihedraAngle
        """
    if self.ordered_aa_ic_list[0].hedra == {}:
        for ric in self.ordered_aa_ic_list:
            ric._create_edra(verbose=verbose)
    if not hasattr(self, 'atomArrayValid'):
        self.build_atomArray()
    if not hasattr(self, 'hedraLen'):
        self.hedraLen = len(self.hedra)
        self.hedraL12 = np.empty(self.hedraLen, dtype=np.float64)
        self.hedraAngle = np.empty(self.hedraLen, dtype=np.float64)
        self.hedraL23 = np.empty(self.hedraLen, dtype=np.float64)
        self.hedraNdx = dict(zip(sorted(self.hedra.keys()), range(len(self.hedra))))
        self.dihedraLen = len(self.dihedra)
        self.dihedraAngle = np.empty(self.dihedraLen)
        self.dihedraAngleRads = np.empty(self.dihedraLen)
        self.dihedraNdx = dict(zip(sorted(self.dihedra.keys()), range(self.dihedraLen)))
    if not hasattr(self, 'hAtoms_needs_update'):
        self.build_edraArrays()