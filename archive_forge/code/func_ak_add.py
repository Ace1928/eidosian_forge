import re
from datetime import date
from io import StringIO
import numpy as np
from Bio.File import as_handle
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.parse_pdb_header import _parse_pdb_header_list
from Bio.PDB.PDBExceptions import PDBException
from Bio.Data.PDBData import protein_letters_1to3
from Bio.PDB.internal_coords import (
from Bio.PDB.ic_data import (
from typing import TextIO, Set, List, Tuple, Union, Optional
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue
from Bio import SeqIO
def ak_add(ek: Tuple, ric: IC_Residue) -> None:
    """Allocate edron key AtomKeys to current residue as appropriate.

        A hedron or dihedron may span a backbone amide bond, this routine
        allocates atoms in the (h/di)edron to the ric residue or saves them
        for a residue yet to be processed.

        :param set ek: AtomKeys in edron
        :param IC_Residue ric: current residue to assign AtomKeys to
        """
    res = ric.residue
    reskl = (str(res.id[1]), None if res.id[2] == ' ' else res.id[2], ric.lc)
    for ak in ek:
        if ak.ric is None:
            sbcic.akset.add(ak)
            if ak.akl[0:3] == reskl:
                ak.ric = ric
                ric.ak_set.add(ak)
            else:
                orphan_aks.add(ak)