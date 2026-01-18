import threading
import time
from taskflow.engines.worker_based import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_execute_task_publish_error(self):
    self.proxy_inst_mock.publish.side_effect = Exception('Woot!')
    ex = self.executor()
    ex._finder._add(self.executor_topic, [self.task.name])
    ex.execute_task(self.task, self.task_uuid, self.task_args)
    expected_calls = [mock.call.Request(self.task, self.task_uuid, 'execute', self.task_args, timeout=self.timeout, result=mock.ANY, failures=mock.ANY), mock.call.request.transition_and_log_error(pr.PENDING, logger=mock.ANY), mock.call.proxy.publish(self.request_inst_mock, self.executor_topic, reply_to=self.executor_uuid, correlation_id=self.task_uuid), mock.call.request.transition_and_log_error(pr.FAILURE, logger=mock.ANY), mock.call.request.set_result(mock.ANY)]
    self.assertEqual(expected_calls, self.master_mock.mock_calls)