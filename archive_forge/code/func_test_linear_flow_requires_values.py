from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_linear_flow_requires_values(self):
    flow = lf.Flow('lf').add(utils.TaskOneArg('task1'), utils.TaskMultiArg('task2'))
    self.assertEqual(set(['x', 'y', 'z']), flow.requires)
    self.assertEqual(set(), flow.provides)