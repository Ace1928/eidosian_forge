from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_ambiguous_provides(self):
    task1 = _task(name='task1', provides=['a', 'b'])
    task2 = _task(name='task2', provides=['a'])
    f = gf.Flow('test')
    f.add(task1, task2)
    self.assertEqual(set(['a', 'b']), f.provides)
    task3 = _task(name='task3', requires=['a'])
    self.assertRaises(exc.AmbiguousDependency, f.add, task3)