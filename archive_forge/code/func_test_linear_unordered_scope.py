from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import scopes as sc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import test
from taskflow.tests import utils as test_utils
def test_linear_unordered_scope(self):
    r = lf.Flow('root')
    r_1 = test_utils.TaskOneReturn('root.1')
    r_2 = test_utils.TaskOneReturn('root.2')
    r.add(r_1, r_2)
    u = uf.Flow('subroot')
    atoms = []
    for i in range(0, 5):
        atoms.append(test_utils.TaskOneReturn('subroot.%s' % i))
    u.add(*atoms)
    r.add(u)
    r_3 = test_utils.TaskOneReturn('root.3')
    r.add(r_3)
    c = compiler.PatternCompiler(r).compile()
    self.assertEqual([], _get_scopes(c, r_1))
    self.assertEqual([['root.1']], _get_scopes(c, r_2))
    for a in atoms:
        self.assertEqual([[], ['root.2', 'root.1']], _get_scopes(c, a))
    scope = _get_scopes(c, r_3)
    self.assertEqual(1, len(scope))
    first_root = 0
    for i, n in enumerate(scope[0]):
        if n.startswith('root.'):
            first_root = i
            break
    first_subroot = 0
    for i, n in enumerate(scope[0]):
        if n.startswith('subroot.'):
            first_subroot = i
            break
    self.assertGreater(first_subroot, first_root)
    self.assertEqual(['root.2', 'root.1'], scope[0][-2:])