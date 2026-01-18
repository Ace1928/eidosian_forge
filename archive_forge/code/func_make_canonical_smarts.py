import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def make_canonical_smarts(subgraph, enumeration_mol, atom_assignment):
    cangen_nodes = get_initial_cangen_nodes(subgraph, enumeration_mol, atom_assignment, True)
    return generate_smarts(cangen_nodes)