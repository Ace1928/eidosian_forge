import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_change_return_value_via_delegate(self):

    def f():
        pass
    mock = create_autospec(f)
    mock.mock.return_value = 1
    self.assertEqual(mock(), 1)