import sys
import os
import pytest
from nipype.pipeline import engine as pe
from nipype.interfaces import base as nib
def test_no_more_memory_than_specified(tmpdir):
    tmpdir.chdir()
    pipe = pe.Workflow(name='pipe')
    n1 = pe.Node(SingleNodeTestInterface(), name='n1', mem_gb=1)
    n2 = pe.Node(SingleNodeTestInterface(), name='n2', mem_gb=1)
    n3 = pe.Node(SingleNodeTestInterface(), name='n3', mem_gb=1)
    n4 = pe.Node(SingleNodeTestInterface(), name='n4', mem_gb=1)
    pipe.connect(n1, 'output1', n2, 'input1')
    pipe.connect(n1, 'output1', n3, 'input1')
    pipe.connect(n2, 'output1', n4, 'input1')
    pipe.connect(n3, 'output1', n4, 'input2')
    n1.inputs.input1 = 1
    max_memory = 0.5
    with pytest.raises(RuntimeError):
        pipe.run(plugin='MultiProc', plugin_args={'memory_gb': max_memory, 'n_procs': 2})