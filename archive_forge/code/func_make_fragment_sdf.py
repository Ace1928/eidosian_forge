import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def make_fragment_sdf(mcs, mol, subgraph, args):
    fragment = subgraph_to_fragment(mol, subgraph)
    Chem.FastFindRings(fragment)
    _copy_sd_tags(mol, fragment)
    if args.save_atom_class_tag is not None:
        output_tag = args.save_atom_class_tag
        atom_classes = get_selected_atom_classes(mol, subgraph.atom_indices)
        if atom_classes is not None:
            fragment.SetProp(output_tag, ' '.join((str(x) for x in atom_classes)))
    _save_other_tags(fragment, fragment, mcs, mol, subgraph, args)
    return _MolToSDBlock(fragment)