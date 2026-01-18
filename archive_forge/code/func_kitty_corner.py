import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def kitty_corner(self):
    if not self.exterior:
        return kitty_corner(self.turns)