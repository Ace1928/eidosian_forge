from copy import deepcopy
from glob import glob
import os
import pytest
from ... import engine as pe
from .test_base import EngineTestInterface
import networkx
def test_mapnode_json(tmpdir):
    """Tests that mapnodes don't generate excess jsons"""
    tmpdir.chdir()
    wd = os.getcwd()
    from nipype import MapNode, Function, Workflow

    def func1(in1):
        return in1 + 1
    n1 = MapNode(Function(input_names=['in1'], output_names=['out'], function=func1), iterfield=['in1'], name='n1')
    n1.inputs.in1 = [1]
    w1 = Workflow(name='test')
    w1.base_dir = wd
    w1.config['execution']['crashdump_dir'] = wd
    w1.add_nodes([n1])
    w1.run()
    n1.inputs.in1 = [2]
    w1.run()
    n1.inputs.in1 = [1]
    eg = w1.run()
    node = list(eg.nodes())[0]
    outjson = glob(os.path.join(node.output_dir(), '_0x*.json'))
    assert len(outjson) == 1
    with open(os.path.join(node.output_dir(), 'test.json'), 'wt') as fp:
        fp.write('dummy file')
    w1.config['execution'].update(**{'stop_on_first_rerun': True})
    w1.run()