import copy
import sys
import warnings
import numpy as np
from Bio.PDB.Entity import DisorderedEntityWrapper
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.vectors import Vector
from Bio.Data import IUPACData
def is_disordered(self):
    """Return the disordered flag (1 if disordered, 0 otherwise)."""
    return self.disordered_flag