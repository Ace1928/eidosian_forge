import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def sleep_on(self, step_num, target, sleep_time):
    self.side_effects[step_num, target] = float(sleep_time)