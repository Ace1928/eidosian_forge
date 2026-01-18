from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def randomize_within_lengths(items):
    by_lens = collections.defaultdict(list)
    for item in items:
        by_lens[len(item)].append(item)
    ans = []
    for length, some_items in sorted(by_lens.items(), reverse=True):
        random.shuffle(some_items)
        ans += some_items
    return ans