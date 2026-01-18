import copy
import sys
import warnings
import numpy as np
from Bio.PDB.Entity import DisorderedEntityWrapper
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.vectors import Vector
from Bio.Data import IUPACData
def set_siguij(self, siguij_array):
    """Set standard deviations of anisotropic temperature factors.

        :param siguij_array: standard deviations of anisotropic temperature factors.
        :type siguij_array: NumPy array (length 6)
        """
    self.siguij_array = siguij_array