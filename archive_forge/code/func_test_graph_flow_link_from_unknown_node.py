from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_link_from_unknown_node(self):
    task1 = _task('task1')
    task2 = _task('task2')
    f = gf.Flow('test').add(task2)
    self.assertRaisesRegex(ValueError, 'Node .* not found to link from', f.link, task1, task2)