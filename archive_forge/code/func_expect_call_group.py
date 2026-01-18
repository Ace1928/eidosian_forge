import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def expect_call_group(self, step_num, targets):
    self.expected_groups.append(set(((step_num, t) for t in targets)))