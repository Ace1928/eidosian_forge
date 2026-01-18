from copy import deepcopy
from glob import glob
import os
import pytest
from ... import engine as pe
from .test_base import EngineTestInterface
import networkx
@pytest.mark.parametrize('iterables, expected, connect', [({'1': {}, '2': dict(input1=lambda: [1, 2]), '3': {}}, (5, 4), ('1-2', '2-3')), ({'1': dict(input1=lambda: [1, 2]), '2': {}, '3': {}}, (5, 4), ('1-3', '2-3')), ({'1': dict(input1=lambda: [1, 2]), '2': dict(input1=lambda: [1, 2]), '3': {}}, (8, 8), ('1-3', '2-3'))])
def test_3mods(iterables, expected, connect):
    pipe = pe.Workflow(name='pipe')
    mod1 = pe.Node(interface=EngineTestInterface(), name='mod1')
    mod2 = pe.Node(interface=EngineTestInterface(), name='mod2')
    mod3 = pe.Node(interface=EngineTestInterface(), name='mod3')
    for nr in ['1', '2', '3']:
        setattr(eval('mod' + nr), 'iterables', iterables[nr])
    if connect == ('1-2', '2-3'):
        pipe.connect([(mod1, mod2, [('output1', 'input2')]), (mod2, mod3, [('output1', 'input2')])])
    elif connect == ('1-3', '2-3'):
        pipe.connect([(mod1, mod3, [('output1', 'input1')]), (mod2, mod3, [('output1', 'input2')])])
    else:
        raise Exception('connect pattern is not implemented yet within the test function')
    pipe._flatgraph = pipe._create_flat_graph()
    pipe._execgraph = pe.generate_expanded_graph(deepcopy(pipe._flatgraph))
    assert len(pipe._execgraph.nodes()) == expected[0]
    assert len(pipe._execgraph.edges()) == expected[1]
    edgenum = sorted([len(pipe._execgraph.in_edges(node)) + len(pipe._execgraph.out_edges(node)) for node in pipe._execgraph.nodes()])
    assert edgenum[0] > 0