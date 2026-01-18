import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def find_extensions(atom_indices, visited_bond_indices, directed_edges):
    internal_bonds = set()
    external_edges = []
    for atom_index in atom_indices:
        for directed_edge in directed_edges[atom_index]:
            if directed_edge.bond_index in visited_bond_indices:
                continue
            if directed_edge.end_atom_index in atom_indices:
                internal_bonds.add(directed_edge.bond_index)
            else:
                external_edges.append(directed_edge)
    return (list(internal_bonds), external_edges)