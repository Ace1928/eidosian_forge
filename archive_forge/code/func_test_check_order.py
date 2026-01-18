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
def test_check_order(self):
    lines = self.make_tag_lines()
    headers = lines[:4]
    rest = lines[4:]
    for perm in permutations(headers):
        perm = list(perm)
        text = b'\n'.join(perm + rest)
        if perm == headers:
            self.assertCheckSucceeds(Tag, text)
        else:
            self.assertCheckFails(Tag, text)