import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def test_error_callback(self):
    check_error_callback(self, try_imports, ['os.doesntexist', 'os.notthiseither'], 2, False)
    check_error_callback(self, try_imports, ['os.doesntexist', 'os.notthiseither', 'os'], 2, True)
    check_error_callback(self, try_imports, ['os.path'], 0, True)