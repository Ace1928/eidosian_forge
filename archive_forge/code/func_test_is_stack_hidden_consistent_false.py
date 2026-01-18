from testtools import TestCase
from testtools.tests.helpers import (
def test_is_stack_hidden_consistent_false(self):
    hide_testtools_stack(False)
    self.assertEqual(False, is_stack_hidden())