import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
def test_itersource_join_source_node(tmpdir):
    """Test join on an input node which has an ``itersource``."""
    tmpdir.chdir()
    wf = pe.Workflow(name='test')
    inputspec = pe.Node(IdentityInterface(fields=['n']), name='inputspec')
    inputspec.iterables = [('n', [1, 2])]
    pre_join1 = pe.Node(IncrementInterface(), name='pre_join1')
    wf.connect(inputspec, 'n', pre_join1, 'input1')
    pre_join2 = pe.Node(ProductInterface(), name='pre_join2')
    pre_join2.itersource = ('inputspec', 'n')
    pre_join2.iterables = ('input1', {1: [3, 4], 2: [5, 6]})
    wf.connect(pre_join1, 'output1', pre_join2, 'input2')
    pre_join3 = pe.Node(IncrementInterface(), name='pre_join3')
    wf.connect(pre_join2, 'output1', pre_join3, 'input1')
    join = pe.JoinNode(IdentityInterface(fields=['vector']), joinsource='pre_join2', joinfield='vector', name='join')
    wf.connect(pre_join3, 'output1', join, 'vector')
    post_join1 = pe.Node(SumInterface(), name='post_join1')
    wf.connect(join, 'vector', post_join1, 'input1')
    result = wf.run()
    assert len(result.nodes()) == 14, 'The number of expanded nodes is incorrect.'
    assert [16, 19] in _sum_operands, 'The join Sum input is incorrect: %s.' % _sum_operands
    assert [7, 9] in _sum_operands, 'The join Sum input is incorrect: %s.' % _sum_operands