import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_bad_start_state(self):
    m = self._create_fsm('unknown', add_start=False)
    r = runners.FiniteRunner(m)
    self.assertRaises(excp.NotFound, r.run, 'unknown')