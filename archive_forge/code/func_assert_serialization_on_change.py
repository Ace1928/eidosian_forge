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
@contextmanager
def assert_serialization_on_change(self, obj, needs_serialization_after_change=True):
    old_id = obj.id
    self.assertFalse(obj._needs_serialization)
    yield obj
    if needs_serialization_after_change:
        self.assertTrue(obj._needs_serialization)
    else:
        self.assertFalse(obj._needs_serialization)
    new_id = obj.id
    self.assertFalse(obj._needs_serialization)
    self.assertNotEqual(old_id, new_id)