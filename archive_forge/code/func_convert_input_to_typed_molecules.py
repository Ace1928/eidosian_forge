import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def convert_input_to_typed_molecules(mols, atom_typer, bond_typer, matchValences, ringMatchesRingOnly):
    typed_mols = []
    for molno, rdmol in enumerate(mols):
        typed_mol = get_typed_molecule(rdmol, atom_typer, bond_typer, matchValences=matchValences, ringMatchesRingOnly=ringMatchesRingOnly)
        typed_mols.append(typed_mol)
    return typed_mols