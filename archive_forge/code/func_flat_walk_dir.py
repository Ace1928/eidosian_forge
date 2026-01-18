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
def flat_walk_dir(dir_to_walk):
    for dirpath, _, filenames in os.walk(dir_to_walk):
        rel_dirpath = os.path.relpath(dirpath, dir_to_walk)
        if not dirpath == dir_to_walk:
            yield rel_dirpath
        for filename in filenames:
            if dirpath == dir_to_walk:
                yield filename
            else:
                yield os.path.join(rel_dirpath, filename)