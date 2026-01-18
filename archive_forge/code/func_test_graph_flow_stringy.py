from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_stringy(self):
    f = gf.Flow('test')
    expected = '"graph_flow.Flow: test(len=0)"'
    self.assertEqual(expected, str(f))
    task1 = _task(name='task1')
    task2 = _task(name='task2')
    task3 = _task(name='task3')
    f = gf.Flow('test')
    f.add(task1, task2, task3)
    expected = '"graph_flow.Flow: test(len=3)"'
    self.assertEqual(expected, str(f))