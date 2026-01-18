import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def find_extension_size(enumeration_mol, known_atoms, exclude_bonds, directed_edges):
    num_remaining_atoms = num_remaining_bonds = 0
    visited_atoms = set(known_atoms)
    visited_bonds = set(exclude_bonds)
    for directed_edge in directed_edges:
        stack = [directed_edge.end_atom_index]
        while stack:
            atom_index = stack.pop()
            for next_edge in enumeration_mol.directed_edges[atom_index]:
                bond_index = next_edge.bond_index
                if bond_index in visited_bonds:
                    continue
                num_remaining_bonds += 1
                visited_bonds.add(bond_index)
                next_atom_index = next_edge.end_atom_index
                if next_atom_index in visited_atoms:
                    continue
                num_remaining_atoms += 1
                visited_atoms.add(next_atom_index)
                stack.append(next_atom_index)
    return (num_remaining_atoms, num_remaining_bonds)