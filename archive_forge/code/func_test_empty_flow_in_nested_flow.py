from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_empty_flow_in_nested_flow(self):
    flow = lf.Flow('lf')
    a = test_utils.ProvidesRequiresTask('a', provides=[], requires=[])
    b = test_utils.ProvidesRequiresTask('b', provides=[], requires=[])
    flow2 = lf.Flow('lf-2')
    c = test_utils.ProvidesRequiresTask('c', provides=[], requires=[])
    d = test_utils.ProvidesRequiresTask('d', provides=[], requires=[])
    empty_flow = gf.Flow('empty')
    flow2.add(c, empty_flow, d)
    flow.add(a, flow2, b)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flow).compile())
    for u, v in [('lf', 'a'), ('a', 'lf-2'), ('lf-2', 'c'), ('c', 'empty'), ('empty[$]', 'd'), ('d', 'lf-2[$]'), ('lf-2[$]', 'b'), ('b', 'lf[$]')]:
        self.assertTrue(g.has_edge(u, v))