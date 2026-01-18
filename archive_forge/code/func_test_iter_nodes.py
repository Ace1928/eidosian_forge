from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_iter_nodes(self):
    task1 = _task('task1', provides=['a'], requires=['c'])
    task2 = _task('task2', provides=['b'], requires=['a'])
    task3 = _task('task3', provides=['c'])
    f1 = gf.Flow('nested')
    f1.add(task3)
    tasks = set([task1, task2, f1])
    f = gf.Flow('test').add(task1, task2, f1)
    for n, data in f.iter_nodes():
        self.assertIn(n, tasks)
        self.assertDictEqual({}, data)