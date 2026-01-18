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
def ak_expand(eLst: List) -> List:
    """Expand AtomKey list with altlocs, all combinatorics."""
    retList = []
    for edron in eLst:
        newList = []
        for ak in edron:
            rslt = ak.ric.split_akl([ak])
            rlst = [r[0] for r in rslt]
            if rlst != []:
                newList.append(rlst)
            else:
                newList.append([ak])
        rslt = ake_recurse(newList)
        for r in rslt:
            retList.append(r)
    return retList