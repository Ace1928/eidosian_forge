from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import scopes as sc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import test
from taskflow.tests import utils as test_utils
def test_nested_prior_linear_begin_middle_end(self):
    r = lf.Flow('root')
    begin_r = test_utils.TaskOneReturn('root.1')
    r.add(begin_r, test_utils.TaskOneReturn('root.2'))
    middle_r = test_utils.TaskOneReturn('root.3')
    r.add(middle_r)
    sub_r = lf.Flow('subroot')
    sub_r.add(test_utils.TaskOneReturn('subroot.1'), test_utils.TaskOneReturn('subroot.2'))
    r.add(sub_r)
    end_r = test_utils.TaskOneReturn('root.4')
    r.add(end_r)
    c = compiler.PatternCompiler(r).compile()
    self.assertEqual([], _get_scopes(c, begin_r))
    self.assertEqual([['root.2', 'root.1']], _get_scopes(c, middle_r))
    self.assertEqual([['subroot.2', 'subroot.1', 'root.3', 'root.2', 'root.1']], _get_scopes(c, end_r))