import math
import sys
import warnings
from collections import Counter
from Bio import BiopythonDeprecationWarning
from Bio.Seq import Seq
def get_residue(self, pos):
    """Return the residue letter at the specified position."""
    return self.pssm[pos][0]