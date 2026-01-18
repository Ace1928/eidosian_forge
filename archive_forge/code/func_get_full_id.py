import copy
import sys
import warnings
import numpy as np
from Bio.PDB.Entity import DisorderedEntityWrapper
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.vectors import Vector
from Bio.Data import IUPACData
def get_full_id(self):
    """Return the full id of the atom.

        The full id of an atom is a tuple used to uniquely identify
        the atom and consists of the following elements:
        (structure id, model id, chain id, residue id, atom name, altloc)
        """
    try:
        return self.parent.get_full_id() + ((self.name, self.altloc),)
    except AttributeError:
        return (None, None, None, None, self.name, self.altloc)