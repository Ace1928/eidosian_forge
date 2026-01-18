from collections import deque
from copy import copy
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionException
def disordered_get_id_list(self):
    """Return a list of id's."""
    return sorted(self.child_dict)