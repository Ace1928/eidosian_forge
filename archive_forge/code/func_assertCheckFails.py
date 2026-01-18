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
def assertCheckFails(self, cls, data):
    obj = cls()

    def do_check():
        obj.set_raw_string(data)
        obj.check()
    self.assertRaises(ObjectFormatException, do_check)