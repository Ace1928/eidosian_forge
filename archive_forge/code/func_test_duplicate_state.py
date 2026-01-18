import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_duplicate_state(self):
    m = self._create_fsm('unknown')
    self.assertRaises(excp.Duplicate, m.add_state, 'unknown')