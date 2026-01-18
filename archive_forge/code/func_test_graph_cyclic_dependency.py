from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_cyclic_dependency(self):
    flow = gf.Flow('g-3-cyclic')
    self.assertRaisesRegex(exceptions.DependencyFailure, '^No path', flow.add, utils.TaskOneArgOneReturn(provides='a', requires=['b']), utils.TaskOneArgOneReturn(provides='b', requires=['c']), utils.TaskOneArgOneReturn(provides='c', requires=['a']))