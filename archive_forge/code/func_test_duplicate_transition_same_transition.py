import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_duplicate_transition_same_transition(self):
    m = self.jumper
    m.add_transition('up', 'down', 'fall')