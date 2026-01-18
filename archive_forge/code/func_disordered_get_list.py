import copy
import sys
import warnings
import numpy as np
from Bio.PDB.Entity import DisorderedEntityWrapper
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.vectors import Vector
from Bio.Data import IUPACData
def disordered_get_list(self):
    """Return list of atom instances.

        Sorts children by altloc (empty, then alphabetical).
        """
    return sorted(self.child_dict.values(), key=lambda a: ord(a.altloc))