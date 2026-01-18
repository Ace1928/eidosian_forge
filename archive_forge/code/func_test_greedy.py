import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_greedy(self):
    cycle = nx_app.greedy_tsp(self.DG, source='D')
    cost = sum((self.DG[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
    validate_solution(cycle, cost, ['D', 'C', 'B', 'A', 'D'], 31.0)
    cycle = nx_app.greedy_tsp(self.DG2, source='D')
    cost = sum((self.DG2[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
    validate_solution(cycle, cost, ['D', 'C', 'B', 'A', 'D'], 78.0)
    cycle = nx_app.greedy_tsp(self.UG, source='D')
    cost = sum((self.UG[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
    validate_solution(cycle, cost, ['D', 'C', 'B', 'A', 'D'], 33.0)
    cycle = nx_app.greedy_tsp(self.UG2, source='D')
    cost = sum((self.UG2[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
    validate_solution(cycle, cost, ['D', 'C', 'A', 'B', 'D'], 27.0)