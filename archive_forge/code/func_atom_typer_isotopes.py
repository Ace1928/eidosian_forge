import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def atom_typer_isotopes(atoms):
    atom_smarts_types = []
    for atom in atoms:
        mass = atom.GetMass()
        int_mass = int(round(mass * 1000))
        if int_mass % 1000 == 0:
            atom_smarts = '%d*' % (int_mass // 1000)
        else:
            atom.SetMass(0.0)
            atom_smarts = '0*'
        atom_smarts_types.append(atom_smarts)
    return atom_smarts_types