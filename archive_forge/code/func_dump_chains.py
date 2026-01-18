from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def dump_chains(self, node):
    chains = []
    for d in self.locals[node]:
        chains.append(str(d))
    return chains