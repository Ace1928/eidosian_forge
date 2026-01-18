import os
from copy import deepcopy
import pytest
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu
from .... import config
from ..utils import (
def test_mapnode_crash2(tmpdir):
    """Test mapnode crash when stop_on_first_crash is False"""
    cwd = os.getcwd()
    node = pe.MapNode(niu.Function(input_names=['WRONG'], output_names=['newstring'], function=dummy_func), iterfield=['WRONG'], name='myfunc')
    node.inputs.WRONG = ['string{}'.format(i) for i in range(3)]
    node.base_dir = tmpdir.strpath
    with pytest.raises(Exception):
        node.run()
    os.chdir(cwd)