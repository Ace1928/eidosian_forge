import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_assert_not_called_message(self):
    m = Mock()
    m(1, 2)
    self.assertRaisesRegex(AssertionError, re.escape('Calls: [call(1, 2)]'), m.assert_not_called)