from collections import defaultdict
from itertools import combinations
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from networkx.utils import not_implemented_for
def build_k_number_dict(kcomps):
    result = {}
    for k, comps in sorted(kcomps.items(), key=itemgetter(0)):
        for comp in comps:
            for node in comp:
                result[node] = k
    return result