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
def test_get_untracked_paths_subdir(self):
    with open(os.path.join(self.repo.path, '.gitignore'), 'w') as f:
        f.write('subdir/\nignored')
    with open(os.path.join(self.repo.path, 'notignored'), 'w') as f:
        f.write('blah\n')
    os.mkdir(os.path.join(self.repo.path, 'subdir'))
    with open(os.path.join(self.repo.path, 'ignored'), 'w') as f:
        f.write('foo')
    with open(os.path.join(self.repo.path, 'subdir', 'ignored'), 'w') as f:
        f.write('foo')
    self.assertEqual({'.gitignore', 'notignored', 'ignored', os.path.join('subdir', '')}, set(porcelain.get_untracked_paths(self.repo.path, self.repo.path, self.repo.open_index())))
    self.assertEqual({'.gitignore', 'notignored'}, set(porcelain.get_untracked_paths(self.repo.path, self.repo.path, self.repo.open_index(), exclude_ignored=True)))