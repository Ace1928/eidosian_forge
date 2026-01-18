from collections import deque
from copy import copy
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionException
def disordered_select(self, id):
    """Select the object with given id as the currently active object.

        Uncaught method calls are forwarded to the selected child object.
        """
    self.selected_child = self.child_dict[id]