from itertools import cycle, islice
import pytest
import networkx as nx
def cyclic_equals(seq1, seq2):
    """Decide whether two sequences are equal up to cyclic permutations.

    For example::

        >>> cyclic_equals("xyz", "zxy")
        True
        >>> cyclic_equals("xyz", "zyx")
        False

    """
    seq2 = tuple(seq2)
    return any((x == tuple(seq2) for x in cycles(seq1)))