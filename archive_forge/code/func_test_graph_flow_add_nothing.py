from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_add_nothing(self):
    f = gf.Flow('test')
    result = f.add()
    self.assertIs(f, result)
    self.assertEqual(0, len(f))