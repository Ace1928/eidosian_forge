import sys
import uuid
import warnings
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Sized
from itertools import chain, tee
import networkx as nx
def create_random_state(random_state=None):
    """Returns a numpy.random.RandomState or numpy.random.Generator instance
    depending on input.

    Parameters
    ----------
    random_state : int or NumPy RandomState or Generator instance, optional (default=None)
        If int, return a numpy.random.RandomState instance set with seed=int.
        if `numpy.random.RandomState` instance, return it.
        if `numpy.random.Generator` instance, return it.
        if None or numpy.random, return the global random number generator used
        by numpy.random.
    """
    import numpy as np
    if random_state is None or random_state is np.random:
        return np.random.mtrand._rand
    if isinstance(random_state, np.random.RandomState):
        return random_state
    if isinstance(random_state, int):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.Generator):
        return random_state
    msg = f'{random_state} cannot be used to create a numpy.random.RandomState or\nnumpy.random.Generator instance'
    raise ValueError(msg)