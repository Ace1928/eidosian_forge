from oslo_utils import reflection
from taskflow.engines.worker_based import types as worker_types
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
def test_multi_different_topic_workers(self):
    finder = worker_types.ProxyWorkerFinder('me', mock.MagicMock(), [])
    added = []
    added.append(finder._add('dummy-topic', [utils.DummyTask]))
    added.append(finder._add('dummy-topic-2', [utils.DummyTask]))
    added.append(finder._add('dummy-topic-3', [utils.NastyTask]))
    self.assertEqual(3, finder.total_workers)
    w = finder.get_worker_for_task(utils.NastyTask)
    self.assertEqual(added[-1][0].identity, w.identity)
    w = finder.get_worker_for_task(utils.DummyTask)
    self.assertIn(w.identity, [w_a[0].identity for w_a in added[0:2]])