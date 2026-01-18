import os
import tempfile
from io import BytesIO
from itertools import chain
from ...objects import hex_to_sha
from ...repo import Repo, check_ref_format
from .utils import CompatTestCase, require_git_version, rmtree_ro, run_git_or_fail
def assertShasMatch(self, expected_shas, actual_shas_iter):
    actual_shas = set()
    for sha in actual_shas_iter:
        obj = self._repo[sha]
        self.assertEqual(sha, obj.id)
        actual_shas.add(sha)
    self.assertEqual(expected_shas, actual_shas)