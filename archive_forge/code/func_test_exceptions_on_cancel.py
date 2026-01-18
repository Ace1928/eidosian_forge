import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_exceptions_on_cancel(self):

    class TestException(Exception):
        pass

    class ExceptionOnExit(Exception):
        pass
    cancelled = []

    def task_func(arg):
        for i in range(4):
            if i > 1:
                raise TestException
            try:
                yield
            except GeneratorExit:
                cancelled.append(arg)
                raise ExceptionOnExit
    tasks = (('A', None), ('B', None), ('C', None))
    deps = dependencies.Dependencies(tasks)
    tg = scheduler.DependencyTaskGroup(deps, task_func)
    task = tg()
    next(task)
    next(task)
    self.assertRaises(TestException, next, task)
    self.assertEqual(len(tasks) - 1, len(cancelled))