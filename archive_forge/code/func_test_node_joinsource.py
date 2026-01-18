import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
def test_node_joinsource(tmpdir):
    """Test setting the joinsource to a Node."""
    tmpdir.chdir()
    inputspec = pe.Node(IdentityInterface(fields=['n']), name='inputspec')
    inputspec.iterables = [('n', [1, 2])]
    join = pe.JoinNode(SetInterface(), joinsource=inputspec, joinfield='input1', name='join')
    assert join.joinsource == inputspec.name, 'The joinsource is not set to the node name.'