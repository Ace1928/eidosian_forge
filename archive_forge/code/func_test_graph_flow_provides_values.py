from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_provides_values(self):
    flow = gf.Flow('gf').add(utils.TaskOneReturn('task1', provides='x'), utils.TaskMultiReturn('task2', provides=['a', 'b', 'c']))
    self.assertEqual(set(), flow.requires)
    self.assertEqual(set(['x', 'a', 'b', 'c']), flow.provides)