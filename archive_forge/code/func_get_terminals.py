import random
import sys
from . import Nodes
def get_terminals(self):
    """Return a list of all terminal nodes."""
    return [i for i in self.all_ids() if self.node(i).succ == []]