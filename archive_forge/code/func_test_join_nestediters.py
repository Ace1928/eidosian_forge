import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
def test_join_nestediters(tmpdir):
    tmpdir.chdir()

    def exponent(x, p):
        return x ** p
    wf = pe.Workflow('wf', base_dir=tmpdir.strpath)
    xs = pe.Node(IdentityInterface(['x']), iterables=[('x', [1, 2])], name='xs')
    ps = pe.Node(IdentityInterface(['p']), iterables=[('p', [3, 4])], name='ps')
    exp = pe.Node(Function(function=exponent), name='exp')
    exp_joinx = pe.JoinNode(Merge(1, ravel_inputs=True), name='exp_joinx', joinsource='xs', joinfield=['in1'])
    exp_joinp = pe.JoinNode(Merge(1, ravel_inputs=True), name='exp_joinp', joinsource='ps', joinfield=['in1'])
    wf.connect([(xs, exp, [('x', 'x')]), (ps, exp, [('p', 'p')]), (exp, exp_joinx, [('out', 'in1')]), (exp_joinx, exp_joinp, [('out', 'in1')])])
    wf.run()