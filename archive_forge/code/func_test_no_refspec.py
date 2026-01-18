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
def test_no_refspec(self):
    outstream = BytesIO()
    errstream = BytesIO()
    porcelain.pull(self.target_path, self.repo.path, outstream=outstream, errstream=errstream)
    with Repo(self.target_path) as r:
        self.assertEqual(r[b'HEAD'].id, self.repo[b'HEAD'].id)