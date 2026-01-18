import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_build_transitions_dct(self):
    space = [{'name': 'down', 'is_terminal': False, 'next_states': {'jump': 'up'}}, {'name': 'up', 'is_terminal': False, 'next_states': {'fall': 'down'}}]
    m = machines.FiniteMachine.build(space)
    m.default_start_state = 'down'
    expected = [('down', 'jump', 'up'), ('up', 'fall', 'down')]
    self.assertEqual(expected, list(m))