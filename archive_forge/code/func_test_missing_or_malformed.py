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
def test_missing_or_malformed(self):
    self.fallback('0 + 0500')
    self.fallback('a +0500')
    self.fallback('1995-11-20T19:12:08')
    self.fallback('1995-11-20T19:12:08-05:')
    self.fallback('1995.11.20')
    self.fallback('11/20/1995')
    self.fallback('20.11.1995')