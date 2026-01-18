from collections import deque
from copy import copy
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionException
def disordered_get(self, id=None):
    """Get the child object associated with id.

        If id is None, the currently selected child is returned.
        """
    if id is None:
        return self.selected_child
    return self.child_dict[id]