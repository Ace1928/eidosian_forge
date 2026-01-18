import contextlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from io import BytesIO, StringIO
from unittest import skipIf
from dulwich import porcelain
from dulwich.tests import TestCase
from ..diff_tree import tree_changes
from ..errors import CommitError
from ..objects import ZERO_SHA, Blob, Tag, Tree
from ..porcelain import CheckoutError
from ..repo import NoIndexPresent, Repo
from ..server import DictBackend
from ..web import make_server, make_wsgi_chain
from .utils import build_commit_graph, make_commit, make_object
def test_add_ignored(self):
    with open(os.path.join(self.repo.path, '.gitignore'), 'w') as f:
        f.write('foo\nsubdir/')
    with open(os.path.join(self.repo.path, 'foo'), 'w') as f:
        f.write('BAR')
    with open(os.path.join(self.repo.path, 'bar'), 'w') as f:
        f.write('BAR')
    os.mkdir(os.path.join(self.repo.path, 'subdir'))
    with open(os.path.join(self.repo.path, 'subdir', 'baz'), 'w') as f:
        f.write('BAZ')
    added, ignored = porcelain.add(self.repo.path, paths=[os.path.join(self.repo.path, 'foo'), os.path.join(self.repo.path, 'bar'), os.path.join(self.repo.path, 'subdir')])
    self.assertIn(b'bar', self.repo.open_index())
    self.assertEqual({'bar'}, set(added))
    self.assertEqual({'foo', os.path.join('subdir', '')}, ignored)