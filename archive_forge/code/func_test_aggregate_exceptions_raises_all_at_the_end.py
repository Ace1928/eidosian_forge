import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_aggregate_exceptions_raises_all_at_the_end(self):

    def run_tasks_with_exceptions(e1=None, e2=None):
        self.aggregate_exceptions = True
        tasks = (('A', None), ('B', None), ('C', None))
        with self._dep_test(*tasks) as track:
            track.expect_call_group(1, ('A', 'B', 'C'))
            track.raise_on(1, 'C', e1)
            track.expect_call_group(2, ('A', 'B'))
            track.raise_on(2, 'B', e2)
            track.expect_call(3, 'A')
    e1 = Exception('e1')
    e2 = Exception('e2')
    exc = self.assertRaises(scheduler.ExceptionGroup, run_tasks_with_exceptions, e1, e2)
    self.assertEqual(set([e1, e2]), set(exc.exceptions))