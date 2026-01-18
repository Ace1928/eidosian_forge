from copy import deepcopy
from glob import glob
import os
import pytest
from ... import engine as pe
from .test_base import EngineTestInterface
import networkx
def test_deep_nested_write_graph_runs(tmpdir):
    tmpdir.chdir()
    for graph in ('orig', 'flat', 'exec', 'hierarchical', 'colored'):
        for simple in (True, False):
            pipe = pe.Workflow(name='pipe')
            parent = pipe
            for depth in range(10):
                sub = pe.Workflow(name='pipe_nest_{}'.format(depth))
                parent.add_nodes([sub])
                parent = sub
            mod1 = pe.Node(interface=EngineTestInterface(), name='mod1')
            parent.add_nodes([mod1])
            try:
                pipe.write_graph(graph2use=graph, simple_form=simple, format='dot')
            except Exception as e:
                assert False, 'Failed to plot {} {} deep graph: {!s}'.format('simple' if simple else 'detailed', graph, e)
            assert os.path.exists('graph.dot') or os.path.exists('graph_detailed.dot')
            try:
                os.remove('graph.dot')
            except OSError:
                pass
            try:
                os.remove('graph_detailed.dot')
            except OSError:
                pass