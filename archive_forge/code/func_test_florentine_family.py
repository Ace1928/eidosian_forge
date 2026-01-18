import random
import pytest
import networkx as nx
def test_florentine_family():
    G = nx.florentine_families_graph()
    indep = nx.maximal_independent_set(G, ['Medici', 'Bischeri'])
    assert set(indep) == {'Medici', 'Bischeri', 'Castellani', 'Pazzi', 'Ginori', 'Lamberteschi'}