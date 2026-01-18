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
def link_residues(ppr: List[Residue], pr: List[Residue]) -> None:
    """Set next and prev links between i-1 and i-2 residues."""
    for p_r in pr:
        pric = p_r.internal_coord
        for p_p_r in ppr:
            ppric = p_p_r.internal_coord
            if p_r.id[0] == ' ':
                if pric not in ppric.rnext:
                    ppric.rnext.append(pric)
            if p_p_r.id[0] == ' ':
                if ppric not in pric.rprev:
                    pric.rprev.append(ppric)