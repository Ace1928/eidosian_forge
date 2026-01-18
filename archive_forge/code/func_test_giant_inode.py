import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_giant_inode(self):
    packed = self.pack((33252, 66571995836, 0, 0, 0, 0, 0, 0, 0, 0))
    self.assertEqual(2147486396, self.unpack_field(packed, 'st_ino'))