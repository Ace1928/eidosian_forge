from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_retry_and_task(self):
    flow = gf.Flow('gf', retry.AlwaysRevert('rt', requires=['x', 'y'], provides=['a', 'b']))
    flow.add(utils.TaskMultiArgOneReturn(rebind=['a', 'x', 'c'], provides=['z']))
    self.assertEqual(set(['x', 'y', 'c']), flow.requires)
    self.assertEqual(set(['a', 'b', 'z']), flow.provides)