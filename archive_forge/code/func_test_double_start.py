import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_double_start(self):
    runner = scheduler.TaskRunner(DummyTask())
    runner.start()
    self.assertRaises(AssertionError, runner.start)