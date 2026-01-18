import warnings
import numpy as np
from Bio.File import as_handle
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.parse_pdb_header import _parse_pdb_header_list
def get_trailer(self):
    """Return the trailer."""
    return self.trailer