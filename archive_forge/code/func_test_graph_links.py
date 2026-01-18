from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_graph_links(self):
    a, b, c, d = test_utils.make_many(4)
    flo = gf.Flow('test')
    flo.add(a, b, c, d)
    flo.link(a, b)
    flo.link(b, c)
    flo.link(c, d)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertEqual(6, len(g))
    self.assertCountEqual(g.edges(data=True), [('test', 'a', {'invariant': True}), ('a', 'b', {'manual': True}), ('b', 'c', {'manual': True}), ('c', 'd', {'manual': True}), ('d', 'test[$]', {'invariant': True})])
    self.assertCountEqual(['test'], g.no_predecessors_iter())
    self.assertCountEqual(['test[$]'], g.no_successors_iter())