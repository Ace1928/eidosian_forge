from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_graph_nested_provides(self):
    a = test_utils.ProvidesRequiresTask('a', provides=[], requires=['x'])
    b = test_utils.ProvidesRequiresTask('b', provides=['x'], requires=[])
    c = test_utils.ProvidesRequiresTask('c', provides=[], requires=[])
    inner_flo = lf.Flow('test2').add(b, c)
    flo = gf.Flow('test').add(a, inner_flo)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertEqual(7, len(g))
    self.assertCountEqual(g.edges(data=True), [('test', 'test2', {'invariant': True}), ('a', 'test[$]', {'invariant': True}), ('test2[$]', 'a', {'reasons': set(['x'])}), ('test2', 'b', {'invariant': True}), ('b', 'c', {'invariant': True}), ('c', 'test2[$]', {'invariant': True})])
    self.assertCountEqual(['test'], g.no_predecessors_iter())
    self.assertCountEqual(['test[$]'], g.no_successors_iter())