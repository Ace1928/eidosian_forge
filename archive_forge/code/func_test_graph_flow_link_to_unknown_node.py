from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_link_to_unknown_node(self):
    task1 = _task('task1')
    task2 = _task('task2')
    f = gf.Flow('test').add(task1)
    self.assertRaisesRegex(ValueError, 'Node .* not found to link to', f.link, task1, task2)