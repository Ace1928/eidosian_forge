import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_nonascii(self):
    self.assertLtPathByDirblock([b'', b'a', b'\xc2\xb5', b'\xc3\xa5', b'a/a', b'a/\xc2\xb5', b'a/\xc3\xa5', b'a/a/a', b'a/a/\xc2\xb5', b'a/a/\xc3\xa5', b'a/\xc2\xb5/a', b'a/\xc2\xb5/\xc2\xb5', b'a/\xc2\xb5/\xc3\xa5', b'a/\xc3\xa5/a', b'a/\xc3\xa5/\xc2\xb5', b'a/\xc3\xa5/\xc3\xa5', b'\xc2\xb5/a', b'\xc2\xb5/\xc2\xb5', b'\xc2\xb5/\xc3\xa5', b'\xc3\xa5/a', b'\xc3\xa5/\xc2\xb5', b'\xc3\xa5/\xc3\xa5'])