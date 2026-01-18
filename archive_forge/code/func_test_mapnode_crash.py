import os
from copy import deepcopy
import pytest
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu
from .... import config
from ..utils import (
def test_mapnode_crash(tmpdir):
    """Test mapnode crash when stop_on_first_crash is True"""
    cwd = os.getcwd()
    node = pe.MapNode(niu.Function(input_names=['WRONG'], output_names=['newstring'], function=dummy_func), iterfield=['WRONG'], name='myfunc')
    node.inputs.WRONG = ['string{}'.format(i) for i in range(3)]
    node.config = deepcopy(config._sections)
    node.config['execution']['stop_on_first_crash'] = True
    node.base_dir = tmpdir.strpath
    with pytest.raises(pe.nodes.NodeExecutionError):
        node.run()
    os.chdir(cwd)