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
def test_add_default_paths(self):
    fullpath = os.path.join(self.repo.path, 'blah')
    with open(fullpath, 'w') as f:
        f.write('\n')
    porcelain.add(repo=self.repo.path, paths=[fullpath])
    porcelain.commit(repo=self.repo.path, message=b'test', author=b'test <email>', committer=b'test <email>')
    with open(os.path.join(self.repo.path, 'foo'), 'w') as f:
        f.write('\n')
    os.mkdir(os.path.join(self.repo.path, 'adir'))
    with open(os.path.join(self.repo.path, 'adir', 'afile'), 'w') as f:
        f.write('\n')
    cwd = os.getcwd()
    try:
        os.chdir(self.repo.path)
        self.assertEqual({'foo', 'blah', 'adir', '.git'}, set(os.listdir('.')))
        self.assertEqual((['foo', os.path.join('adir', 'afile')], set()), porcelain.add(self.repo.path))
    finally:
        os.chdir(cwd)
    index = self.repo.open_index()
    self.assertEqual(sorted(index), [b'adir/afile', b'blah', b'foo'])