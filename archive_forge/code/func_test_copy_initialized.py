import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_copy_initialized(self):
    j = self.jumper.copy()
    self.assertIsNone(j.current_state)
    r = runners.FiniteRunner(self.jumper)
    for i, transition in enumerate(r.run_iter('jump')):
        if i == 4:
            break
    self.assertIsNone(j.current_state)
    self.assertIsNotNone(self.jumper.current_state)