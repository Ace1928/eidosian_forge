import random
import pytest
import networkx as nx
from networkx.utils import arbitrary_element, graphs_equal
@pytest.mark.parametrize('strings', (['a', 'to', 'tea', 'ted', 'ten', 'i', 'in', 'inn'], ['ab', 'abs', 'ad'], ['ab', 'abs', 'ad', ''], ['distant', 'disparaging', 'distant', 'diamond', 'ruby']))
def test_implementations_consistent(strings):
    """Ensure results are consistent between prefix_tree implementations."""
    assert graphs_equal(nx.prefix_tree(strings), nx.prefix_tree_recursive(strings))