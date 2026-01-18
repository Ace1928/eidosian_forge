from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_empty_flow_in_graph_flow_linkage(self):
    flow = gf.Flow('lf')
    a = test_utils.ProvidesRequiresTask('a', provides=[], requires=[])
    b = test_utils.ProvidesRequiresTask('b', provides=[], requires=[])
    empty_flow = lf.Flow('empty')
    flow.add(a, empty_flow, b)
    flow.link(a, b)
    compilation = compiler.PatternCompiler(flow).compile()
    g = compilation.execution_graph
    self.assertTrue(g.has_edge(a, b))
    self.assertTrue(g.has_edge(flow, a))
    self.assertTrue(g.has_edge(flow, empty_flow))