import sys
import uuid
import warnings
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Sized
from itertools import chain, tee
import networkx as nx
def dict_to_numpy_array(d, mapping=None):
    """Convert a dictionary of dictionaries to a numpy array
    with optional mapping."""
    try:
        return _dict_to_numpy_array2(d, mapping)
    except (AttributeError, TypeError):
        return _dict_to_numpy_array1(d, mapping)