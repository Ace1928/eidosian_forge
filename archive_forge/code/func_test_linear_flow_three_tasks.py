from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_linear_flow_three_tasks(self):
    task1 = _task(name='task1')
    task2 = _task(name='task2')
    task3 = _task(name='task3')
    f = lf.Flow('test').add(task1, task2, task3)
    self.assertEqual(3, len(f))
    self.assertEqual([task1, task2, task3], list(f))
    self.assertEqual([(task1, task2, {'invariant': True}), (task2, task3, {'invariant': True})], list(f.iter_links()))