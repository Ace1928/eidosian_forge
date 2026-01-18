import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def test_None_is_default_alternative(self):
    e = self.assertRaises(ImportError, try_imports, ['doesntexist', 'noreally'])
    self.assertThat(str(e), Equals('Could not import any of: doesntexist, noreally'))