import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def task_func(arg):
    for i in range(4):
        if i > 1:
            raise TestException
        try:
            yield
        except GeneratorExit:
            cancelled.append(arg)
            raise ExceptionOnExit