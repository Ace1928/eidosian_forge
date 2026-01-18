import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def get_canonical_bondtype_counts(typed_mols):
    overall_counts = defaultdict(list)
    for typed_mol in typed_mols:
        bondtype_counts = get_counts(typed_mol.canonical_bondtypes)
        for k, v in bondtype_counts.items():
            overall_counts[k].append(v)
    return overall_counts