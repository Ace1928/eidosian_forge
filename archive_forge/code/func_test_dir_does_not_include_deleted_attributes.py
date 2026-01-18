import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_dir_does_not_include_deleted_attributes(self):
    mock = Mock()
    mock.child.return_value = 1
    self.assertIn('child', dir(mock))
    del mock.child
    self.assertNotIn('child', dir(mock))