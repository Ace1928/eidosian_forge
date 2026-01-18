import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
def avg_sim(s):
    return sum((newsim[w][x] for w, x in s)) / len(s) if s else 0.0