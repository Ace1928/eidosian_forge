import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
@patch('os.path')
def patched(mock_path):
    patch.stopall()
    self.assertIs(os.path, mock_path)
    self.assertIs(os.unlink, unlink)
    self.assertIs(os.chdir, chdir)