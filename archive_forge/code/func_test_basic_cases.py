import itertools
import pytest
import networkx as nx
def test_basic_cases(self):

    def check_basic_case(graph_func, n_nodes, strategy, interchange):
        graph = graph_func()
        coloring = nx.coloring.greedy_color(graph, strategy=strategy, interchange=interchange)
        assert verify_length(coloring, n_nodes)
        assert verify_coloring(graph, coloring)
    for graph_func, n_nodes in BASIC_TEST_CASES.items():
        for interchange in [True, False]:
            for strategy in ALL_STRATEGIES:
                check_basic_case(graph_func, n_nodes, strategy, False)
                if strategy not in INTERCHANGE_INVALID:
                    check_basic_case(graph_func, n_nodes, strategy, True)