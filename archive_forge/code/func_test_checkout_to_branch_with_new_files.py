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
def test_checkout_to_branch_with_new_files(self):
    porcelain.checkout_branch(self.repo, b'uni')
    sub_directory = os.path.join(self.repo.path, 'sub1')
    os.mkdir(sub_directory)
    for index in range(5):
        _commit_file_with_content(self.repo, 'new_file_' + str(index + 1), 'Some content\n')
        _commit_file_with_content(self.repo, os.path.join('sub1', 'new_file_' + str(index + 10)), 'Good content\n')
    status = list(porcelain.status(self.repo))
    self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], []], status)
    porcelain.checkout_branch(self.repo, b'master')
    self.assertEqual(b'master', porcelain.active_branch(self.repo))
    status = list(porcelain.status(self.repo))
    self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], []], status)
    porcelain.checkout_branch(self.repo, b'uni')
    self.assertEqual(b'uni', porcelain.active_branch(self.repo))
    status = list(porcelain.status(self.repo))
    self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], []], status)