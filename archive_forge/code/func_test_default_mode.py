import io
import os
import shutil
import sys
import tempfile
from dulwich.tests import SkipTest, TestCase
from ..file import FileLocked, GitFile, _fancy_rename
def test_default_mode(self):
    f = GitFile(self.path('foo'))
    self.assertEqual(b'foo contents', f.read())
    f.close()