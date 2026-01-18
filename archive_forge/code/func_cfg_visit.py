import gast as ast
from collections import defaultdict
from functools import reduce
from pythran.analyses import Aliases, CFG
from pythran.intrinsic import Intrinsic
from pythran.passmanager import ModuleAnalysis
from pythran.interval import Interval, IntervalTuple, UNKNOWN_RANGE
from pythran.tables import MODULES, attributes
def cfg_visit(self, node, skip=None):
    successors = [node]
    visited = set() if skip is None else skip.copy()
    while successors:
        successor = successors.pop()
        if successor in visited:
            continue
        visited.add(successor)
        nexts = self.visit(successor)
        if nexts:
            successors.extend((n for n in nexts if n is not CFG.NIL))