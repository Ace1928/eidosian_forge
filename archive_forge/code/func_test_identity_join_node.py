import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
def test_identity_join_node(tmpdir):
    """Test an IdentityInterface join."""
    global _sum_operands
    _sum_operands = []
    tmpdir.chdir()
    wf = pe.Workflow(name='test')
    inputspec = pe.Node(IdentityInterface(fields=['n']), name='inputspec')
    inputspec.iterables = [('n', [1, 2, 3])]
    pre_join1 = pe.Node(IncrementInterface(), name='pre_join1')
    wf.connect(inputspec, 'n', pre_join1, 'input1')
    join = pe.JoinNode(IdentityInterface(fields=['vector']), joinsource='inputspec', joinfield='vector', name='join')
    wf.connect(pre_join1, 'output1', join, 'vector')
    post_join1 = pe.Node(SumInterface(), name='post_join1')
    wf.connect(join, 'vector', post_join1, 'input1')
    result = wf.run()
    assert len(result.nodes()) == 5, 'The number of expanded nodes is incorrect.'
    assert _sum_operands[0] == [2, 3, 4], 'The join Sum input is incorrect: %s.' % _sum_operands[0]