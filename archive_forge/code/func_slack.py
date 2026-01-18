from collections import Counter
from itertools import combinations, repeat
import networkx as nx
from networkx.utils import not_implemented_for
def slack(v, w):
    return dualvar[v] + dualvar[w] - 2 * G[v][w].get(weight, 1)