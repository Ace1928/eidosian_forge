import os
import pytest
from nipype.interfaces import utility
import nipype.pipeline.engine as pe
def test_aux_connect_function(tmpdir):
    """This tests execution nodes with multiple inputs and auxiliary
    function inside the Workflow connect function.
    """
    tmpdir.chdir()
    wf = pe.Workflow(name='test_workflow')

    def _gen_tuple(size):
        return [1] * size

    def _sum_and_sub_mul(a, b, c):
        return ((a + b) * c, (a - b) * c)

    def _inc(x):
        return x + 1
    params = pe.Node(utility.IdentityInterface(fields=['size', 'num']), name='params')
    params.inputs.num = 42
    params.inputs.size = 1
    gen_tuple = pe.Node(utility.Function(input_names=['size'], output_names=['tuple'], function=_gen_tuple), name='gen_tuple')
    ssm = pe.Node(utility.Function(input_names=['a', 'b', 'c'], output_names=['sum', 'sub'], function=_sum_and_sub_mul), name='sum_and_sub_mul')
    split = pe.Node(utility.Split(splits=[1, 1], squeeze=True), name='split')
    wf.connect([(params, gen_tuple, [(('size', _inc), 'size')]), (params, ssm, [(('num', _inc), 'c')]), (gen_tuple, split, [('tuple', 'inlist')]), (split, ssm, [(('out1', _inc), 'a'), ('out2', 'b')])])
    wf.run()