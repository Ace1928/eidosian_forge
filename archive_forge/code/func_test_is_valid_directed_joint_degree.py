import time
from networkx.algorithms.assortativity import degree_mixing_dict
from networkx.generators import gnm_random_graph, powerlaw_cluster_graph
from networkx.generators.joint_degree_seq import (
def test_is_valid_directed_joint_degree():
    in_degrees = [0, 1, 1, 2]
    out_degrees = [1, 1, 1, 1]
    nkk = {1: {1: 2, 2: 2}}
    assert is_valid_directed_joint_degree(in_degrees, out_degrees, nkk)
    nkk = {1: {1: 1.5, 2: 2.5}}
    assert not is_valid_directed_joint_degree(in_degrees, out_degrees, nkk)
    nkk = {1: {1: 2, 2: 1}}
    assert not is_valid_directed_joint_degree(in_degrees, out_degrees, nkk)
    out_degrees = [1, 1, 1]
    nkk = {1: {1: 2, 2: 2}}
    assert not is_valid_directed_joint_degree(in_degrees, out_degrees, nkk)
    in_degrees = [0, 1, 2]
    assert not is_valid_directed_joint_degree(in_degrees, out_degrees, nkk)