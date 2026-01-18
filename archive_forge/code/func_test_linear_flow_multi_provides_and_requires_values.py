from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_linear_flow_multi_provides_and_requires_values(self):
    flow = lf.Flow('lf').add(utils.TaskMultiArgMultiReturn('task1', rebind=['a', 'b', 'c'], provides=['x', 'y', 'q']), utils.TaskMultiArgMultiReturn('task2', provides=['i', 'j', 'k']))
    self.assertEqual(set(['a', 'b', 'c', 'z']), flow.requires)
    self.assertEqual(set(['x', 'y', 'q', 'i', 'j', 'k']), flow.provides)