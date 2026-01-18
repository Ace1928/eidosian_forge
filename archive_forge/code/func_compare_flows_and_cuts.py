import pytest
import networkx as nx
from networkx.algorithms.flow import (
def compare_flows_and_cuts(G, s, t, solnFlows, solnValue, capacity='capacity'):
    for flow_func in flow_funcs:
        errmsg = f'Assertion failed in function: {flow_func.__name__}'
        R = flow_func(G, s, t, capacity)
        flow_value = R.graph['flow_value']
        flow_dict = build_flow_dict(G, R)
        assert flow_value == solnValue, errmsg
        validate_flows(G, s, t, flow_dict, solnValue, capacity, flow_func)
        cut_value, partition = nx.minimum_cut(G, s, t, capacity=capacity, flow_func=flow_func)
        validate_cuts(G, s, t, solnValue, partition, capacity, flow_func)