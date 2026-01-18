from itertools import combinations
import networkx as nx
from networkx.algorithms.simple_paths import is_simple_path as is_path
from networkx.utils import arbitrary_element, not_implemented_for, py_random_state
def index_satisfying(iterable, condition):
    """Returns the index of the first element in `iterable` that
    satisfies the given condition.

    If no such element is found (that is, when the iterable is
    exhausted), this returns the length of the iterable (that is, one
    greater than the last index of the iterable).

    `iterable` must not be empty. If `iterable` is empty, this
    function raises :exc:`ValueError`.

    """
    for i, x in enumerate(iterable):
        if condition(x):
            return i
    try:
        return i + 1
    except NameError as err:
        raise ValueError('iterable must be non-empty') from err