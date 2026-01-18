from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_linear_nested(self):
    a, b, c, d = test_utils.make_many(4)
    flo = lf.Flow('test')
    flo.add(a, b)
    inner_flo = uf.Flow('test2')
    inner_flo.add(c, d)
    flo.add(inner_flo)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertEqual(8, len(g))
    sub_g = g.subgraph(['a', 'b'])
    self.assertFalse(sub_g.has_edge('b', 'a'))
    self.assertTrue(sub_g.has_edge('a', 'b'))
    self.assertEqual({'invariant': True}, sub_g.get_edge_data('a', 'b'))
    sub_g = g.subgraph(['c', 'd'])
    self.assertEqual(0, sub_g.number_of_edges())
    self.assertTrue(g.has_edge('b', 'test2'))
    self.assertTrue(g.has_edge('test2', 'c'))
    self.assertTrue(g.has_edge('test2', 'd'))