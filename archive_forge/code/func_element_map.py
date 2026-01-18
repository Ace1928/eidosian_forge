import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def element_map(partition):
    return {x: P for P in partition for x in P}