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
def pdb_date(datestr: str) -> str:
    """Convert yyyy-mm-dd date to dd-month-yy."""
    if datestr:
        m = re.match('(\\d{4})-(\\d{2})-(\\d{2})', datestr)
        if m:
            mo = ['XXX', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'][int(m.group(2))]
            datestr = m.group(3) + '-' + mo + '-' + m.group(1)[-2:]
    return datestr