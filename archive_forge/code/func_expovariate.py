import sys
import uuid
import warnings
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Sized
from itertools import chain, tee
import networkx as nx
def expovariate(self, scale):
    return self._rng.exponential(1 / scale)