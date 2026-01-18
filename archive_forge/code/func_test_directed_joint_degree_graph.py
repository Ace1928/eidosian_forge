import time
from networkx.algorithms.assortativity import degree_mixing_dict
from networkx.generators import gnm_random_graph, powerlaw_cluster_graph
from networkx.generators.joint_degree_seq import (
def test_directed_joint_degree_graph(n=15, m=100, ntimes=1000):
    for _ in range(ntimes):
        g = gnm_random_graph(n, m, None, directed=True)
        in_degrees = list(dict(g.in_degree()).values())
        out_degrees = list(dict(g.out_degree()).values())
        nkk = degree_mixing_dict(g)
        G = directed_joint_degree_graph(in_degrees, out_degrees, nkk)
        assert in_degrees == list(dict(G.in_degree()).values())
        assert out_degrees == list(dict(G.out_degree()).values())
        assert nkk == degree_mixing_dict(G)