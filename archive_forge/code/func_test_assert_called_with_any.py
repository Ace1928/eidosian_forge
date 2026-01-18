import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_assert_called_with_any(self):
    m = MagicMock()
    m(MagicMock())
    m.assert_called_with(mock.ANY)