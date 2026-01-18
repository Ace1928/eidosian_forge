import futurist
from futurist import waiters
from oslo_utils import uuidutils
from taskflow.engines.action_engine import executor as base_executor
from taskflow.engines.worker_based import endpoint
from taskflow.engines.worker_based import executor as worker_executor
from taskflow.engines.worker_based import server as worker_server
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
from taskflow.utils import threading_utils
def test_execution_failure_pipeline(self):
    task_classes = [test_utils.TaskWithFailure]
    executor, server = self._start_components(task_classes)
    t = test_utils.TaskWithFailure()
    progress_callback = lambda *args, **kwargs: None
    f = executor.execute_task(t, uuidutils.generate_uuid(), {}, progress_callback=progress_callback)
    waiters.wait_for_any([f])
    action, result = f.result()
    self.assertIsInstance(result, failure.Failure)
    self.assertEqual(RuntimeError, result.check(RuntimeError))
    self.assertEqual(base_executor.EXECUTED, action)