import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def atom_typer_elements(atoms):
    return [_atom_smarts_no_aromaticity[atom.GetAtomicNum()] for atom in atoms]