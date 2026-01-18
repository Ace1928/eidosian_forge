import copy
import sys
import warnings
import numpy as np
from Bio.PDB.Entity import DisorderedEntityWrapper
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.vectors import Vector
from Bio.Data import IUPACData
def flag_disorder(self):
    """Set the disordered flag to 1.

        The disordered flag indicates whether the atom is disordered or not.
        """
    self.disordered_flag = 1