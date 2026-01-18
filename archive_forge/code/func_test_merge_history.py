import glob
import locale
import os
import shutil
import stat
import sys
import tempfile
import warnings
from dulwich import errors, objects, porcelain
from dulwich.tests import TestCase, skipIf
from ..config import Config
from ..errors import NotGitRepository
from ..object_store import tree_lookup_path
from ..repo import (
from .utils import open_repo, setup_warning_catcher, tear_down_repo
import sys
from dulwich.repo import Repo
def test_merge_history(self):
    r = self.open_repo('simple_merge.git')
    shas = [e.commit.id for e in r.get_walker()]
    self.assertEqual(shas, [b'5dac377bdded4c9aeb8dff595f0faeebcc8498cc', b'ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd', b'4cffe90e0a41ad3f5190079d7c8f036bde29cbe6', b'60dacdc733de308bb77bb76ce0fb0f9b44c9769e', b'0d89f20333fbb1d2f3a94da77f4981373d8f4310'])