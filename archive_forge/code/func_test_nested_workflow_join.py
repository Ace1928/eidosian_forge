import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
def test_nested_workflow_join(tmpdir):
    """Test collecting join inputs within a nested workflow"""
    tmpdir.chdir()

    def nested_wf(i, name='smallwf'):
        inputspec = pe.Node(IdentityInterface(fields=['n']), name='inputspec')
        inputspec.iterables = [('n', i)]
        pre_join = pe.Node(IncrementInterface(), name='pre_join')
        join = pe.JoinNode(IdentityInterface(fields=['n']), joinsource='inputspec', joinfield='n', name='join')
        wf = pe.Workflow(name='wf_%d' % i[0])
        wf.connect(inputspec, 'n', pre_join, 'input1')
        wf.connect(pre_join, 'output1', join, 'n')
        return wf
    meta_wf = pe.Workflow(name='meta', base_dir='.')
    for i in [[1, 3], [2, 4]]:
        mini_wf = nested_wf(i)
        meta_wf.add_nodes([mini_wf])
    result = meta_wf.run()
    assert len(result.nodes()) == 6, 'The number of expanded nodes is incorrect.'