from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import scopes as sc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import test
from taskflow.tests import utils as test_utils
def test_graph_linear_scope(self):
    r = gf.Flow('root')
    r_1 = test_utils.TaskOneReturn('root.1')
    r_2 = test_utils.TaskOneReturn('root.2')
    r.add(r_1, r_2)
    r.link(r_1, r_2)
    s = lf.Flow('subroot')
    s_1 = test_utils.TaskOneReturn('subroot.1')
    s_2 = test_utils.TaskOneReturn('subroot.2')
    s.add(s_1, s_2)
    r.add(s)
    t = gf.Flow('subroot2')
    t_1 = test_utils.TaskOneReturn('subroot2.1')
    t_2 = test_utils.TaskOneReturn('subroot2.2')
    t.add(t_1, t_2)
    t.link(t_1, t_2)
    r.add(t)
    r.link(s, t)
    c = compiler.PatternCompiler(r).compile()
    self.assertEqual([], _get_scopes(c, r_1))
    self.assertEqual([['root.1']], _get_scopes(c, r_2))
    self.assertEqual([], _get_scopes(c, s_1))
    self.assertEqual([['subroot.1']], _get_scopes(c, s_2))
    self.assertEqual([[], ['subroot.2', 'subroot.1']], _get_scopes(c, t_1))
    self.assertEqual([['subroot2.1'], ['subroot.2', 'subroot.1']], _get_scopes(c, t_2))