import copy
import sys
import warnings
import numpy as np
from Bio.PDB.Entity import DisorderedEntityWrapper
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.vectors import Vector
from Bio.Data import IUPACData
def set_sigatm(self, sigatm_array):
    """Set standard deviation of atomic parameters.

        The standard deviation of atomic parameters consists
        of 3 positional, 1 B factor and 1 occupancy standard
        deviation.

        :param sigatm_array: standard deviations of atomic parameters.
        :type sigatm_array: NumPy array (length 5)
        """
    self.sigatm_array = sigatm_array