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
def process_hedron(a1: str, a2: str, a3: str, l12: str, ang: str, l23: str, ric: IC_Residue) -> Tuple:
    """Create Hedron on current (sbcic) Chain.internal_coord."""
    ek = (akcache(a1), akcache(a2), akcache(a3))
    atmNdx = AtomKey.fields.atm
    accpt = IC_Residue.accept_atoms
    if not all((ek[i].akl[atmNdx] in accpt for i in range(3))):
        return
    hl12[ek] = float(l12)
    ha[ek] = float(ang)
    hl23[ek] = float(l23)
    sbcic.hedra[ek] = ric.hedra[ek] = h = Hedron(ek)
    h.cic = sbcic
    ak_add(ek, ric)
    return ek