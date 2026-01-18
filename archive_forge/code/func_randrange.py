import sys
import uuid
import warnings
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Sized
from itertools import chain, tee
import networkx as nx
def randrange(self, a, b=None):
    import numpy as np
    if isinstance(self._rng, np.random.Generator):
        return self._rng.integers(a, b)
    return self._rng.randint(a, b)