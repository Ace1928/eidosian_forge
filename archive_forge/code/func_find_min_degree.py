import itertools
from collections import defaultdict, deque
import networkx as nx
from networkx.utils import arbitrary_element, py_random_state
def find_min_degree():
    return next((d for d in itertools.count(lbound) if d in degrees))