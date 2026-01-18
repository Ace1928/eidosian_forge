import os
import re
import shutil
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..ignore import (
from ..repo import Repo
def test_included(self):
    filter = IgnoreFilter([b'a.c', b'b.c'])
    self.assertTrue(filter.is_ignored(b'a.c'))
    self.assertIs(None, filter.is_ignored(b'c.c'))
    self.assertEqual([Pattern(b'a.c')], list(filter.find_matching(b'a.c')))
    self.assertEqual([], list(filter.find_matching(b'c.c')))