import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_can_add_exceptions_after_init(self):
    ex = Exception()
    exception_group = scheduler.ExceptionGroup()
    exception_group.exceptions.append(ex)
    self.assertIn(ex, exception_group.exceptions)