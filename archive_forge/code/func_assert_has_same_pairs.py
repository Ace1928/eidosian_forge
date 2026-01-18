from itertools import chain, combinations, product
import pytest
import networkx as nx
@staticmethod
def assert_has_same_pairs(d1, d2):
    for a, b in ((min(pair), max(pair)) for pair in chain(d1, d2)):
        assert get_pair(d1, a, b) == get_pair(d2, a, b)