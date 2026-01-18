import sys
import uuid
import warnings
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Sized
from itertools import chain, tee
import networkx as nx
def create_py_random_state(random_state=None):
    """Returns a random.Random instance depending on input.

    Parameters
    ----------
    random_state : int or random number generator or None (default=None)
        If int, return a random.Random instance set with seed=int.
        if random.Random instance, return it.
        if None or the `random` package, return the global random number
        generator used by `random`.
        if np.random package, return the global numpy random number
        generator wrapped in a PythonRandomInterface class.
        if np.random.RandomState or np.random.Generator instance, return it
        wrapped in PythonRandomInterface
        if a PythonRandomInterface instance, return it
    """
    import random
    try:
        import numpy as np
        if random_state is np.random:
            return PythonRandomInterface(np.random.mtrand._rand)
        if isinstance(random_state, (np.random.RandomState, np.random.Generator)):
            return PythonRandomInterface(random_state)
        if isinstance(random_state, PythonRandomInterface):
            return random_state
    except ImportError:
        pass
    if random_state is None or random_state is random:
        return random._inst
    if isinstance(random_state, random.Random):
        return random_state
    if isinstance(random_state, int):
        return random.Random(random_state)
    msg = f'{random_state} cannot be used to generate a random.Random instance'
    raise ValueError(msg)