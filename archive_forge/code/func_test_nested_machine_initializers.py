import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_nested_machine_initializers(self):
    dialer, _number_calling = self._make_phone_dialer()
    queried_for = []

    def init_with(nested_machine):
        queried_for.append(nested_machine)
        return None
    dialer.initialize(nested_start_state_fetcher=init_with)
    self.assertEqual(1, len(queried_for))