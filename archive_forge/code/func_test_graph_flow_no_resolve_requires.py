from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_graph_flow_no_resolve_requires(self):
    task1 = _task(name='task1', provides=['a', 'b', 'c'])
    task2 = _task(name='task2', requires=['a', 'b'])
    f = gf.Flow('test')
    f.add(task1, task2, resolve_requires=False)
    self.assertEqual(set(['a', 'b']), f.requires)