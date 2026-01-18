from collections import Counter
from itertools import combinations, repeat
import networkx as nx
from networkx.utils import not_implemented_for
def leaves(self):
    stack = [*self.childs]
    while stack:
        t = stack.pop()
        if isinstance(t, Blossom):
            stack.extend(t.childs)
        else:
            yield t