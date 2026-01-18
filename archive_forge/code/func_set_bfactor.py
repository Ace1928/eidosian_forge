import copy
import sys
import warnings
import numpy as np
from Bio.PDB.Entity import DisorderedEntityWrapper
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.vectors import Vector
from Bio.Data import IUPACData
def set_bfactor(self, bfactor):
    """Set isotroptic B factor."""
    self.bfactor = bfactor