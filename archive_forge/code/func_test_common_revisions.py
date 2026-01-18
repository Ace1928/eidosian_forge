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
def test_common_revisions(self):
    """This test demonstrates that ``find_common_revisions()`` actually
        returns common heads, not revisions; dulwich already uses
        ``find_common_revisions()`` in such a manner (see
        ``Repo.find_objects()``).
        """
    expected_shas = {b'60dacdc733de308bb77bb76ce0fb0f9b44c9769e'}
    r_base = self.open_repo('simple_merge.git')
    r1_dir = self.mkdtemp()
    self.addCleanup(shutil.rmtree, r1_dir)
    r1_commits = [b'ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd', b'60dacdc733de308bb77bb76ce0fb0f9b44c9769e', b'0d89f20333fbb1d2f3a94da77f4981373d8f4310']
    r2_dir = self.mkdtemp()
    self.addCleanup(shutil.rmtree, r2_dir)
    r2_commits = [b'4cffe90e0a41ad3f5190079d7c8f036bde29cbe6', b'60dacdc733de308bb77bb76ce0fb0f9b44c9769e', b'0d89f20333fbb1d2f3a94da77f4981373d8f4310']
    r1 = Repo.init_bare(r1_dir)
    for c in r1_commits:
        r1.object_store.add_object(r_base.get_object(c))
    r1.refs[b'HEAD'] = r1_commits[0]
    r2 = Repo.init_bare(r2_dir)
    for c in r2_commits:
        r2.object_store.add_object(r_base.get_object(c))
    r2.refs[b'HEAD'] = r2_commits[0]
    shas = r2.object_store.find_common_revisions(r1.get_graph_walker())
    self.assertEqual(set(shas), expected_shas)
    shas = r1.object_store.find_common_revisions(r2.get_graph_walker())
    self.assertEqual(set(shas), expected_shas)