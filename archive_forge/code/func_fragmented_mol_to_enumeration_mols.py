import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def fragmented_mol_to_enumeration_mols(typed_mol, minNumAtoms=2):
    if minNumAtoms < 2:
        raise ValueError('minNumAtoms must be at least 2')
    fragments = []
    for atom_indices in Chem.GetMolFrags(typed_mol.rdmol):
        if len(atom_indices) < minNumAtoms:
            continue
        typed_fragment = get_typed_fragment(typed_mol, atom_indices)
        rdmol = typed_fragment.rdmol
        atoms = []
        for atom, orig_atom, atom_smarts_type in zip(rdmol.GetAtoms(), typed_fragment.orig_atoms, typed_fragment.atom_smarts_types):
            bond_indices = [bond.GetIdx() for bond in atom.GetBonds()]
            atom_smarts = '[' + atom_smarts_type + ']'
            atoms.append(Atom(atom, atom_smarts, bond_indices, orig_atom.IsInRing()))
        directed_edges = defaultdict(list)
        bonds = []
        for bond_index, (bond, orig_bond, bond_smarts, canonical_bondtype) in enumerate(zip(rdmol.GetBonds(), typed_fragment.orig_bonds, typed_fragment.bond_smarts_types, typed_fragment.canonical_bondtypes)):
            atom_indices = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            bonds.append(Bond(bond, bond_smarts, canonical_bondtype, atom_indices, orig_bond.IsInRing()))
            directed_edges[atom_indices[0]].append(DirectedEdge(bond_index, atom_indices[1]))
            directed_edges[atom_indices[1]].append(DirectedEdge(bond_index, atom_indices[0]))
        fragment = EnumerationMolecule(rdmol, atoms, bonds, dict(directed_edges))
        fragments.append(fragment)
    fragments.sort(key=lambda fragment: len(fragment.atoms), reverse=True)
    return fragments