import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_mock_call_not_equal_leaf(self):
    m = Mock()
    m.foo().something()
    self.assertNotEqual(m.mock_calls[1], call.foo().different())
    self.assertEqual(m.mock_calls[0], call.foo())