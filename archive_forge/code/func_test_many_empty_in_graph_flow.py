from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_many_empty_in_graph_flow(self):
    flo = gf.Flow('root')
    a = test_utils.ProvidesRequiresTask('a', provides=[], requires=[])
    flo.add(a)
    b = lf.Flow('b')
    b_0 = test_utils.ProvidesRequiresTask('b.0', provides=[], requires=[])
    b_1 = lf.Flow('b.1')
    b_2 = lf.Flow('b.2')
    b_3 = test_utils.ProvidesRequiresTask('b.3', provides=[], requires=[])
    b.add(b_0, b_1, b_2, b_3)
    flo.add(b)
    c = lf.Flow('c')
    c_0 = lf.Flow('c.0')
    c_1 = lf.Flow('c.1')
    c_2 = lf.Flow('c.2')
    c.add(c_0, c_1, c_2)
    flo.add(c)
    d = test_utils.ProvidesRequiresTask('d', provides=[], requires=[])
    flo.add(d)
    flo.link(b, d)
    flo.link(a, d)
    flo.link(c, d)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertTrue(g.has_edge('root', 'a'))
    self.assertTrue(g.has_edge('root', 'b'))
    self.assertTrue(g.has_edge('root', 'c'))
    self.assertTrue(g.has_edge('b.0', 'b.1'))
    self.assertTrue(g.has_edge('b.1[$]', 'b.2'))
    self.assertTrue(g.has_edge('b.2[$]', 'b.3'))
    self.assertTrue(g.has_edge('c.0[$]', 'c.1'))
    self.assertTrue(g.has_edge('c.1[$]', 'c.2'))
    self.assertTrue(g.has_edge('a', 'd'))
    self.assertTrue(g.has_edge('b[$]', 'd'))
    self.assertTrue(g.has_edge('c[$]', 'd'))
    self.assertEqual(20, len(g))