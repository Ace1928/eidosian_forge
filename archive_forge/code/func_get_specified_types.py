import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def get_specified_types(rdmol, atom_types, ringMatchesRingOnly):
    raise NotImplementedError('not tested!')
    rdmol = copy.copy(rdmol)
    atom_smarts_types = []
    atoms = list(mol.GetAtoms())
    for atom, atom_type in zip(atoms, atom_types):
        atom.SetAtomicNum(0)
        atom.SetMass(atom_type)
        atom_term = '%d*' % (atom_type,)
        if ringMatchesRingOnly:
            if atom.IsInRing():
                atom_term += 'R'
            else:
                atom_term += '!R'
        atom_smarts_types.append('[' + atom_term + ']')
    bonds = list(rdmol.GetBonds())
    bond_smarts_types = get_bond_smarts_types(mol, bonds, ringMatchesRingOnly)
    canonical_bondtypes = get_canonical_bondtypes(mol, bonds, atom_smarts_types, bond_smarts_types)
    return TypedMolecule(mol, atoms, bonds, atom_smarts_types, bond_smarts_types, canonical_bondtypes)