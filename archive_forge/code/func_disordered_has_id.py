from collections import deque
from copy import copy
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionException
def disordered_has_id(self, id):
    """Check if there is an object present associated with this id."""
    return id in self.child_dict