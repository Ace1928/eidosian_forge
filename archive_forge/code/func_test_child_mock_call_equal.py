import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_child_mock_call_equal(self):
    m = Mock()
    result = m()
    result.wibble()
    self.assertEqual(m.mock_calls, [call(), call().wibble()])
    self.assertEqual(result.mock_calls, [call.wibble()])