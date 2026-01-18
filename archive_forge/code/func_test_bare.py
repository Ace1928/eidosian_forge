import os
import tempfile
from io import BytesIO
from itertools import chain
from ...objects import hex_to_sha
from ...repo import Repo, check_ref_format
from .utils import CompatTestCase, require_git_version, rmtree_ro, run_git_or_fail
def test_bare(self):
    self.assertFalse(self._repo.bare)
    self.assertTrue(os.path.isfile(os.path.join(self._repo.path, '.git')))