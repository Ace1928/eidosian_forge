from copy import deepcopy
from glob import glob
import os
import pytest
from ... import engine as pe
from .test_base import EngineTestInterface
import networkx
def test_io_subclass():
    """Ensure any io subclass allows dynamic traits"""
    from nipype.interfaces.io import IOBase
    from nipype.interfaces.base import DynamicTraitedSpec

    class TestKV(IOBase):
        _always_run = True
        output_spec = DynamicTraitedSpec

        def _list_outputs(self):
            outputs = {}
            outputs['test'] = 1
            outputs['foo'] = 'bar'
            return outputs
    wf = pe.Workflow('testkv')

    def testx2(test):
        return test * 2
    kvnode = pe.Node(TestKV(), name='testkv')
    from nipype.interfaces.utility import Function
    func = pe.Node(Function(input_names=['test'], output_names=['test2'], function=testx2), name='func')
    exception_not_raised = True
    try:
        wf.connect(kvnode, 'test', func, 'test')
    except Exception as e:
        if 'Module testkv has no output called test' in e:
            exception_not_raised = False
    assert exception_not_raised