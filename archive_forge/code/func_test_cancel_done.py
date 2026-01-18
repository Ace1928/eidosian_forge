import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_cancel_done(self):
    task = DummyTask(1)
    task.do_step = mock.Mock(return_value=None)
    runner = scheduler.TaskRunner(task)
    self.assertFalse(runner.started())
    runner.start()
    self.assertTrue(runner.started())
    self.assertTrue(runner.step())
    self.assertTrue(runner.done())
    runner.cancel()
    self.assertTrue(runner.done())
    self.assertTrue(runner.step())
    task.do_step.assert_called_once_with(1)
    self.mock_sleep.assert_not_called()