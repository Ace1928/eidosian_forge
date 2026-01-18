import os
from copy import deepcopy
import pytest
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu
from .... import config
from ..utils import (
def test_mapnode_crash3(tmpdir):
    """Test mapnode crash when mapnode is embedded in a workflow"""
    tmpdir.chdir()
    node = pe.MapNode(niu.Function(input_names=['WRONG'], output_names=['newstring'], function=dummy_func), iterfield=['WRONG'], name='myfunc')
    node.inputs.WRONG = ['string{}'.format(i) for i in range(3)]
    wf = pe.Workflow('testmapnodecrash')
    wf.add_nodes([node])
    wf.base_dir = tmpdir.strpath
    wf.config['execution']['crashdump_dir'] = os.getcwd()
    with pytest.raises(RuntimeError):
        wf.run(plugin='Linear')