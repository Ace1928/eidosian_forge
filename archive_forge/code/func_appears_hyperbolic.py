import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def appears_hyperbolic(M):
    acceptable = ['all tetrahedra positively oriented', 'contains negatively oriented tetrahedra']
    return M.solution_type() in acceptable and M.volume() > 1.0