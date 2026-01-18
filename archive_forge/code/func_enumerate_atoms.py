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
def enumerate_atoms(entity):
    """Ensure all atoms in entity have serial_number set."""
    while entity.get_parent():
        entity = entity.get_parent()
    if 'S' == entity.level:
        for mdl in entity:
            _enumerate_entity_atoms(mdl)
    else:
        _enumerate_entity_atoms(entity)