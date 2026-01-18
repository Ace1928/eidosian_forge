import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def bond_typer_bondtypes(bonds):
    bond_smarts_types = []
    for bond in bonds:
        bond_term = bond.GetSmarts()
        if not bond_term:
            if bond.GetIsAromatic():
                bond_term = ':'
            else:
                bond_term = '-'
        bond_smarts_types.append(bond_term)
    return bond_smarts_types