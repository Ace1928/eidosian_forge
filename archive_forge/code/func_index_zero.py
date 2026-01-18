import random
import itertools
from ast import literal_eval
from Bio.Phylo import BaseTree
from Bio.Align import MultipleSeqAlignment
def index_zero(self):
    """Return a list of positions where the element is '0'."""
    return [i for i, n in enumerate(self) if n == '0']