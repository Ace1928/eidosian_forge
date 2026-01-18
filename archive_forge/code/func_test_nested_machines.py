import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_nested_machines(self):
    dialer, _number_calling = self._make_phone_dialer()
    self.assertEqual(1, len(dialer.nested_machines))