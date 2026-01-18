import os
import tempfile
from io import BytesIO
from itertools import chain
from ...objects import hex_to_sha
from ...repo import Repo, check_ref_format
from .utils import CompatTestCase, require_git_version, rmtree_ro, run_git_or_fail
def create_new_worktree(self, repo_dir, branch):
    """Create a new worktree using git-worktree.

        Args:
          repo_dir: The directory of the main working tree.
          branch: The branch or commit to checkout in the new worktree.

        Returns: The path to the new working tree.
        """
    temp_dir = tempfile.mkdtemp()
    run_git_or_fail(['worktree', 'add', temp_dir, branch], cwd=repo_dir)
    self.addCleanup(rmtree_ro, temp_dir)
    return temp_dir