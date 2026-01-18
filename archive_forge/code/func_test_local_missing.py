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
def test_local_missing(self):
    """Pushing a new branch."""
    outstream = BytesIO()
    errstream = BytesIO()
    clone_path = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, clone_path)
    target_repo = porcelain.init(clone_path)
    target_repo.close()
    self.assertRaises(porcelain.Error, porcelain.push, self.repo, clone_path, b'HEAD:refs/heads/master', outstream=outstream, errstream=errstream)