from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_empty_flow_in_linear_flow(self):
    flo = lf.Flow('lf')
    a = test_utils.ProvidesRequiresTask('a', provides=[], requires=[])
    b = test_utils.ProvidesRequiresTask('b', provides=[], requires=[])
    empty_flo = gf.Flow('empty')
    flo.add(a, empty_flo, b)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertCountEqual(g.edges(), [('lf', 'a'), ('a', 'empty'), ('empty', 'empty[$]'), ('empty[$]', 'b'), ('b', 'lf[$]')])