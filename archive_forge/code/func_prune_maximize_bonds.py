import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def prune_maximize_bonds(subgraph, mol, num_remaining_atoms, num_remaining_bonds, best_sizes):
    num_atoms = len(subgraph.atom_indices)
    num_bonds = len(subgraph.bond_indices)
    best_num_atoms, best_num_bonds = best_sizes
    diff_bonds = num_bonds + num_remaining_bonds - best_num_bonds
    if diff_bonds < 0:
        return True
    elif diff_bonds == 0:
        diff_atoms = num_atoms + num_remaining_atoms - best_num_atoms
        if diff_atoms <= 0:
            return True
    return False