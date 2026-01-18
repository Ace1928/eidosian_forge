from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_graph_dependencies(self):
    a = test_utils.ProvidesRequiresTask('a', provides=['x'], requires=[])
    b = test_utils.ProvidesRequiresTask('b', provides=[], requires=['x'])
    flo = gf.Flow('test').add(a, b)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertEqual(4, len(g))
    self.assertCountEqual(g.edges(data=True), [('test', 'a', {'invariant': True}), ('a', 'b', {'reasons': set(['x'])}), ('b', 'test[$]', {'invariant': True})])
    self.assertCountEqual(['test'], g.no_predecessors_iter())
    self.assertCountEqual(['test[$]'], g.no_successors_iter())