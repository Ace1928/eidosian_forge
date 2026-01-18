import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_copy_states(self):
    c = self._create_fsm('down', add_start=False)
    self.assertEqual(0, len(c.states))
    d = c.copy()
    c.add_state('up')
    c.add_state('down')
    self.assertEqual(2, len(c.states))
    self.assertEqual(0, len(d.states))