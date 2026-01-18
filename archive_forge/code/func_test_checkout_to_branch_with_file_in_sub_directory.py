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
def test_checkout_to_branch_with_file_in_sub_directory(self):
    sub_directory = os.path.join(self.repo.path, 'sub1', 'sub2')
    os.makedirs(sub_directory)
    sub_directory_file = os.path.join(sub_directory, 'neu')
    with open(sub_directory_file, 'w') as f:
        f.write('new message\n')
    porcelain.add(self.repo, paths=[sub_directory_file])
    porcelain.commit(self.repo, message=b'add ' + sub_directory_file.encode(), committer=b'Jane <jane@example.com>', author=b'John <john@example.com>')
    status = list(porcelain.status(self.repo))
    self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], []], status)
    self.assertTrue(os.path.isdir(sub_directory))
    self.assertTrue(os.path.isdir(os.path.dirname(sub_directory)))
    porcelain.checkout_branch(self.repo, b'uni')
    status = list(porcelain.status(self.repo))
    self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], []], status)
    self.assertFalse(os.path.isdir(sub_directory))
    self.assertFalse(os.path.isdir(os.path.dirname(sub_directory)))
    porcelain.checkout_branch(self.repo, b'master')
    self.assertTrue(os.path.isdir(sub_directory))
    self.assertTrue(os.path.isdir(os.path.dirname(sub_directory)))