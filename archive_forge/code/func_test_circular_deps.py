import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_circular_deps(self):
    d = dependencies.Dependencies([('first', 'second'), ('second', 'third'), ('third', 'first')])
    self.assertRaises(exception.CircularDependencyException, scheduler.DependencyTaskGroup, d)