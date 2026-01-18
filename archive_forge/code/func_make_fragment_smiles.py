import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def make_fragment_smiles(mcs, mol, subgraph, args=None):
    fragment = subgraph_to_fragment(mol, subgraph)
    new_smiles = Chem.MolToSmiles(fragment)
    return '%s %s\n' % (new_smiles, mol.GetProp('_Name'))