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
def test_checkout_remote_branch_then_master_then_remote_branch_again(self):
    target_repo = self._checkout_remote_branch()
    self.assertEqual(b'foo', porcelain.active_branch(target_repo))
    _commit_file_with_content(target_repo, 'bar', 'something\n')
    self.assertTrue(os.path.isfile(os.path.join(target_repo.path, 'bar')))
    porcelain.checkout_branch(target_repo, b'master')
    self.assertEqual(b'master', porcelain.active_branch(target_repo))
    self.assertFalse(os.path.isfile(os.path.join(target_repo.path, 'bar')))
    porcelain.checkout_branch(target_repo, b'origin/foo')
    self.assertEqual(b'foo', porcelain.active_branch(target_repo))
    self.assertTrue(os.path.isfile(os.path.join(target_repo.path, 'bar')))
    target_repo.close()