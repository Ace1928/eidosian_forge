import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def generate_smarts(cangen_nodes):
    start_index = 0
    best_rank = cangen_nodes[0].rank
    for i, node in enumerate(cangen_nodes):
        if node.rank < best_rank:
            best_rank = node.rank
            start_index = i
        node.outgoing_edges.sort(key=lambda edge: edge.other_node.rank)
    visited_atoms = [0] * len(cangen_nodes)
    closure_bonds = set()
    stack = []
    atom_idx = start_index
    stack.extend(reversed(cangen_nodes[atom_idx].outgoing_edges))
    visited_atoms[atom_idx] = True
    while stack:
        edge = stack.pop()
        if visited_atoms[edge.other_node_idx]:
            closure_bonds.add(edge.bond_index)
        else:
            visited_atoms[edge.other_node_idx] = 1
            for next_edge in reversed(cangen_nodes[edge.other_node_idx].outgoing_edges):
                if next_edge.other_node_idx == edge.from_atom_index:
                    continue
                stack.append(next_edge)
    available_closures = _available_closures[:]
    unclosed_closures = {}
    smiles_terms = []
    stack = [(0, (start_index, -1))]
    while stack:
        action, data = stack.pop()
        if action == 0:
            while 1:
                num_neighbors = 0
                atom_idx, prev_bond_idx = data
                smiles_terms.append(cangen_nodes[atom_idx].atom_smarts)
                outgoing_edges = cangen_nodes[atom_idx].outgoing_edges
                for outgoing_edge in outgoing_edges:
                    bond_idx = outgoing_edge.bond_index
                    if bond_idx in closure_bonds:
                        if bond_idx not in unclosed_closures:
                            closure = heappop(available_closures)
                            smiles_terms.append(get_closure_label(outgoing_edge.bond_smarts, closure))
                            unclosed_closures[bond_idx] = closure
                        else:
                            closure = unclosed_closures[bond_idx]
                            smiles_terms.append(get_closure_label(outgoing_edge.bond_smarts, closure))
                            heappush(available_closures, closure)
                            del unclosed_closures[bond_idx]
                    else:
                        if bond_idx == prev_bond_idx:
                            continue
                        if num_neighbors == 0:
                            data = (outgoing_edge.other_node_idx, bond_idx)
                            bond_smarts = outgoing_edge.bond_smarts
                        else:
                            if num_neighbors == 1:
                                stack.append((0, data))
                                stack.append((1, bond_smarts))
                            stack.append((3, None))
                            stack.append((0, (outgoing_edge.other_node_idx, bond_idx)))
                            stack.append((4, outgoing_edge.bond_smarts))
                        num_neighbors += 1
                if num_neighbors != 1:
                    break
                smiles_terms.append(bond_smarts)
        elif action == 1:
            smiles_terms.append(data)
            continue
        elif action == 3:
            smiles_terms.append(')')
        elif action == 4:
            smiles_terms.append('(' + data)
        else:
            raise AssertionError
    return ''.join(smiles_terms)