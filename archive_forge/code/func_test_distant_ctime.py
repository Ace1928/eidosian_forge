import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_distant_ctime(self):
    packed = self.pack((33252, 0, 0, 0, 0, 0, 0, 0, 0, 64060588800.0))
    self.assertEqual(3931046656, self.unpack_field(packed, 'st_ctime'))