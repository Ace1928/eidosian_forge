import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
def test_name_prefix_join(tmpdir):
    tmpdir.chdir()

    def sq(x):
        return x ** 2
    wf = pe.Workflow('wf', base_dir=tmpdir.strpath)
    square = pe.Node(Function(function=sq), name='square')
    square.iterables = [('x', [1, 2])]
    square_join = pe.JoinNode(Merge(1, ravel_inputs=True), name='square_join', joinsource='square', joinfield=['in1'])
    wf.connect(square, 'out', square_join, 'in1')
    wf.run()