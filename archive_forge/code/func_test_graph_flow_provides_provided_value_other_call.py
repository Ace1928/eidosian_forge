from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_provides_provided_value_other_call(self):
    flow = gf.Flow('gf')
    flow.add(utils.TaskOneReturn('task1', provides='x'))
    flow.add(utils.TaskOneReturn('task2', provides='x'))
    self.assertEqual(set(['x']), flow.provides)