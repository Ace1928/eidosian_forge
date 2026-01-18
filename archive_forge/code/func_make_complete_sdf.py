import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def make_complete_sdf(mcs, mol, subgraph, args):
    fragment = copy.copy(mol)
    _copy_sd_tags(mol, fragment)
    if args.save_atom_indices_tag is not None:
        output_tag = args.save_atom_indices_tag
        s = ' '.join((str(index) for index in subgraph.atom_indices))
        fragment.SetProp(output_tag, s)
    _save_other_tags(fragment, subgraph_to_fragment(mol, subgraph), mcs, mol, subgraph, args)
    return _MolToSDBlock(fragment)