from collections import deque
from copy import copy
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionException
def detach_child(self, id):
    """Remove a child."""
    child = self.child_dict[id]
    child.detach_parent()
    del self.child_dict[id]
    self.child_list.remove(child)