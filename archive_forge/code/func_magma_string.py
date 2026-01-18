from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
def magma_string(self):
    gens = sorted((self.alphabet[g] for g in self.generators))
    ans = 'Group<' + ', '.join(gens) + ' | '
    ans += ', '.join((R.verbose_string() for R in self.relators))
    return ans + '>'