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
def get_sha_file(self, cls, base, sha):
    dir = os.path.join(os.path.dirname(__file__), '..', '..', 'testdata', base)
    return cls.from_path(hex_to_filename(dir, sha))