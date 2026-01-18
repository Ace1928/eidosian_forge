from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_dependency_cycle(self):
    task1 = _task('task1', provides=['a'], requires=['c'])
    task2 = _task('task2', provides=['b'], requires=['a'])
    task3 = _task('task3', provides=['c'], requires=['b'])
    f = gf.Flow('test').add(task1, task2)
    self.assertRaises(exc.DependencyFailure, f.add, task3)