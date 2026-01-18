import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def restore_isotopes(mol):
    try:
        isotopes = _isotope_dict[mol]
    except KeyError:
        raise ValueError('no isotopes to restore')
    set_isotopes(mol, isotopes)