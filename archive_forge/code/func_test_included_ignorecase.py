import os
import re
import shutil
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..ignore import (
from ..repo import Repo
def test_included_ignorecase(self):
    filter = IgnoreFilter([b'a.c', b'b.c'], ignorecase=False)
    self.assertTrue(filter.is_ignored(b'a.c'))
    self.assertFalse(filter.is_ignored(b'A.c'))
    filter = IgnoreFilter([b'a.c', b'b.c'], ignorecase=True)
    self.assertTrue(filter.is_ignored(b'a.c'))
    self.assertTrue(filter.is_ignored(b'A.c'))
    self.assertTrue(filter.is_ignored(b'A.C'))