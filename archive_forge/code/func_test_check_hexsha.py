import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
def test_check_hexsha(self):
    check_hexsha(a_sha, 'failed to check good sha')
    self.assertRaises(ObjectFormatException, check_hexsha, b'1' * 39, 'sha too short')
    self.assertRaises(ObjectFormatException, check_hexsha, b'1' * 41, 'sha too long')
    self.assertRaises(ObjectFormatException, check_hexsha, b'x' * 40, 'invalid characters')