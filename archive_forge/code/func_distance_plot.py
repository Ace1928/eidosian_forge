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
def distance_plot(self, filter: Optional[Union[np.ndarray, None]]=None) -> np.ndarray:
    """Generate 2D distance plot from atomArray.

        Default is to calculate distances for all atoms.  To generate the
        classic C-alpha distance plot, pass a boolean mask array like::

            atmNameNdx = internal_coords.AtomKey.fields.atm
            CaSelect = [
                atomArrayIndex.get(k)
                for k in atomArrayIndex.keys()
                if k.akl[atmNameNdx] == "CA"
            ]
            plot = cic.distance_plot(CaSelect)

        Alternatively, this will select all backbone atoms::

            backboneSelect = [
                atomArrayIndex.get(k)
                for k in atomArrayIndex.keys()
                if k.is_backbone()
            ]

        :param [bool] filter: restrict atoms for calculation

        .. seealso::
            :meth:`.distance_to_internal_coordinates`, which requires the
            default all atom distance plot.

        """
    if filter is None:
        atomSet = self.atomArray
    else:
        atomSet = self.atomArray[filter]
    return np.linalg.norm(atomSet[:, None, :] - atomSet[None, :, :], axis=-1)