import pytest
import networkx as nx
from networkx.algorithms.connectivity.kcomponents import (
def list_of_sets_equal(result, solution):
    assert {frozenset(s) for s in result} == {frozenset(s) for s in solution}