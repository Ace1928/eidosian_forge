import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_can_be_initialized_with_a_list_of_exceptions(self):
    ex1 = Exception('ex 1')
    ex2 = Exception('ex 2')
    exception_group = scheduler.ExceptionGroup([ex1, ex2])
    self.assertIn(ex1, exception_group.exceptions)
    self.assertIn(ex2, exception_group.exceptions)