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
def test_hard_head(self):
    fullpath = os.path.join(self.repo.path, 'foo')
    with open(fullpath, 'w') as f:
        f.write('BAR')
    porcelain.add(self.repo.path, paths=[fullpath])
    porcelain.commit(self.repo.path, message=b'Some message', committer=b'Jane <jane@example.com>', author=b'John <john@example.com>')
    with open(os.path.join(self.repo.path, 'foo'), 'wb') as f:
        f.write(b'OOH')
    porcelain.reset(self.repo, 'hard', b'HEAD')
    index = self.repo.open_index()
    changes = list(tree_changes(self.repo, index.commit(self.repo.object_store), self.repo[b'HEAD'].tree))
    self.assertEqual([], changes)