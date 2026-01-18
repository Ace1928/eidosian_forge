import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_build_transitions(self):
    space = [machines.State('down', is_terminal=False, next_states={'jump': 'up'}), machines.State('up', is_terminal=False, next_states={'fall': 'down'})]
    m = machines.FiniteMachine.build(space)
    m.default_start_state = 'down'
    expected = [('down', 'jump', 'up'), ('up', 'fall', 'down')]
    self.assertEqual(expected, list(m))