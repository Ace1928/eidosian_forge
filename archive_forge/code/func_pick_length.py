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
def pick_length(self, ak_spec: Union[str, BKT]) -> Tuple[Optional[List['Hedron']], Optional[BKT]]:
    """Get list of hedra containing specified atom pair.

        :param ak_spec:
            - tuple of two AtomKeys
            - string: two atom names separated by ':', e.g. 'N:CA' with
              optional position specifier relative to self, e.g. '-1C:N' for
              preceding peptide bond.  Position specifiers are -1, 0, 1.

        The following are equivalent::

            ric = r.internal_coord
            print(
                r,
                ric.get_length("0C:1N"),
            )
            print(
                r,
                None
                if not ric.rnext
                else ric.get_length((ric.rak("C"), ric.rnext[0].rak("N"))),
            )

        If atom not found on current residue then will look on rprev[0] to
        handle cases like Gly N:CA.  For finer control please access
        `IC_Chain.hedra` directly.

        :return: list of hedra containing specified atom pair as tuples of
                AtomKeys
        """
    rlst: List[Hedron] = []
    if isinstance(ak_spec, str):
        ak_spec = cast(BKT, self._get_ak_tuple(ak_spec))
    if ak_spec is None:
        return (None, None)
    for hed_key, hed_val in self.hedra.items():
        if all((ak in hed_key for ak in ak_spec)):
            rlst.append(hed_val)
    for rp in self.rprev:
        for hed_key, hed_val in rp.hedra.items():
            if all((ak in hed_key for ak in ak_spec)):
                rlst.append(hed_val)
    return (rlst, ak_spec)