import os
import re
import shutil
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..ignore import (
from ..repo import Repo
def test_include_exclude_include(self):
    filter = IgnoreFilter([b'a.c', b'!a.c', b'a.c'])
    self.assertTrue(filter.is_ignored(b'a.c'))
    self.assertEqual([Pattern(b'a.c'), Pattern(b'!a.c'), Pattern(b'a.c')], list(filter.find_matching(b'a.c')))