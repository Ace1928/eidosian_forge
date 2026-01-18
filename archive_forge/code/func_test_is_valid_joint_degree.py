import time
from networkx.algorithms.assortativity import degree_mixing_dict
from networkx.generators import gnm_random_graph, powerlaw_cluster_graph
from networkx.generators.joint_degree_seq import (
def test_is_valid_joint_degree():
    """Tests for conditions that invalidate a joint degree dict"""
    joint_degrees = {1: {4: 1}, 2: {2: 2, 3: 2, 4: 2}, 3: {2: 2, 4: 1}, 4: {1: 1, 2: 2, 3: 1}}
    assert is_valid_joint_degree(joint_degrees)
    joint_degrees_1 = {1: {4: 1.5}, 2: {2: 2, 3: 2, 4: 2}, 3: {2: 2, 4: 1}, 4: {1: 1.5, 2: 2, 3: 1}}
    assert not is_valid_joint_degree(joint_degrees_1)
    joint_degrees_2 = {1: {4: 1}, 2: {2: 2, 3: 2, 4: 3}, 3: {2: 2, 4: 1}, 4: {1: 1, 2: 3, 3: 1}}
    assert not is_valid_joint_degree(joint_degrees_2)
    joint_degrees_3 = {1: {4: 2}, 2: {2: 2, 3: 2, 4: 2}, 3: {2: 2, 4: 1}, 4: {1: 2, 2: 2, 3: 1}}
    assert not is_valid_joint_degree(joint_degrees_3)
    joint_degrees_5 = {1: {1: 9}}
    assert not is_valid_joint_degree(joint_degrees_5)