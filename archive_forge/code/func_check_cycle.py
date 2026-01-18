from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def check_cycle(self, g, c, es, cache, source, original_c, length_bound, chordless):
    if length_bound is not None and len(c) > length_bound:
        raise RuntimeError(f'computed cycle {original_c} exceeds length bound {length_bound}')
    if source == 'computed':
        if es in cache:
            raise RuntimeError(f'computed cycle {original_c} has already been found!')
        else:
            cache[es] = tuple(original_c)
    elif es in cache:
        cache.pop(es)
    else:
        raise RuntimeError(f'expected cycle {original_c} was not computed')
    if not all((g.has_edge(*e) for e in es)):
        raise RuntimeError(f'{source} claimed cycle {original_c} is not a cycle of g')
    if chordless and len(g.subgraph(c).edges) > len(c):
        raise RuntimeError(f'{source} cycle {original_c} is not chordless')