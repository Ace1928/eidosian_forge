import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_cancel_grace_period_not_started(self):
    task = DummyTask(1)
    task.do_step = mock.Mock(return_value=None)
    runner = scheduler.TaskRunner(task)
    self.assertFalse(runner.started())
    runner.cancel(grace_period=0.5)
    self.assertTrue(runner.done())
    task.do_step.assert_not_called()
    self.mock_sleep.assert_not_called()