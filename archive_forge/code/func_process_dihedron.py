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
def process_dihedron(a1: str, a2: str, a3: str, a4: str, dangle: str, ric: IC_Residue) -> Set:
    """Create Dihedron on current Chain.internal_coord."""
    ek = (akcache(a1), akcache(a2), akcache(a3), akcache(a4))
    atmNdx = AtomKey.fields.atm
    accpt = IC_Residue.accept_atoms
    if not all((ek[i].akl[atmNdx] in accpt for i in range(4))):
        return
    dangle = float(dangle)
    dangle = dangle if dangle <= 180.0 else dangle - 360.0
    dangle = dangle if dangle >= -180.0 else dangle + 360.0
    da[ek] = float(dangle)
    sbcic.dihedra[ek] = ric.dihedra[ek] = d = Dihedron(ek)
    d.cic = sbcic
    if not quick:
        hedra_check(ek, ric)
    ak_add(ek, ric)
    return ek