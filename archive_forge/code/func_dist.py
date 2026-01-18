import math
import random
from itertools import combinations
import pytest
import networkx as nx
def dist(x, y):
    return sum((abs(a - b) for a, b in zip(x, y)))