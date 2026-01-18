from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_graph_nested_requires(self):
    a = test_utils.ProvidesRequiresTask('a', provides=['x'], requires=[])
    b = test_utils.ProvidesRequiresTask('b', provides=[], requires=[])
    c = test_utils.ProvidesRequiresTask('c', provides=[], requires=['x'])
    inner_flo = lf.Flow('test2').add(b, c)
    flo = gf.Flow('test').add(a, inner_flo)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertEqual(7, len(g))
    self.assertCountEqual(g.edges(data=True), [('test', 'a', {'invariant': True}), ('test2', 'b', {'invariant': True}), ('a', 'test2', {'reasons': set(['x'])}), ('b', 'c', {'invariant': True}), ('c', 'test2[$]', {'invariant': True}), ('test2[$]', 'test[$]', {'invariant': True})])
    self.assertCountEqual(['test'], list(g.no_predecessors_iter()))
    self.assertCountEqual(['test[$]'], list(g.no_successors_iter()))