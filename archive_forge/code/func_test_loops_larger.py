import pytest
import networkx as nx
def test_loops_larger(self):
    g = nx.DiGraph()
    edges = [('entry', 'exit'), ('entry', '1'), ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '6'), ('6', 'exit'), ('6', '2'), ('5', '3'), ('4', '4')]
    g.add_edges_from(edges)
    df = nx.dominance_frontiers(g, 'entry')
    answer = {'entry': set(), '1': {'exit'}, '2': {'exit', '2'}, '3': {'exit', '3', '2'}, '4': {'exit', '4', '3', '2'}, '5': {'exit', '3', '2'}, '6': {'exit', '2'}, 'exit': set()}
    for n in df:
        assert set(df[n]) == set(answer[n])