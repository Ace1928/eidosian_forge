import os
import tempfile
from io import BytesIO
from itertools import chain
from ...objects import hex_to_sha
from ...repo import Repo, check_ref_format
from .utils import CompatTestCase, require_git_version, rmtree_ro, run_git_or_fail
def test_git_worktree_config(self):
    """Test that git worktree config parsing matches the git CLI's behavior."""
    require_git_version((2, 7, 0))
    test_name = 'Jelmer'
    test_email = 'jelmer@apache.org'
    run_git_or_fail(['config', 'user.name', test_name], cwd=self._repo.path)
    run_git_or_fail(['config', 'user.email', test_email], cwd=self._repo.path)
    worktree_cfg = self._worktree_repo.get_config()
    main_cfg = self._repo.get_config()
    self.assertEqual(worktree_cfg, main_cfg)
    for c in [worktree_cfg, main_cfg]:
        self.assertEqual(test_name.encode(), c.get((b'user',), b'name'))
        self.assertEqual(test_email.encode(), c.get((b'user',), b'email'))
    output_name = run_git_or_fail(['config', 'user.name'], cwd=self._mainworktree_repo.path).decode().rstrip('\n')
    output_email = run_git_or_fail(['config', 'user.email'], cwd=self._mainworktree_repo.path).decode().rstrip('\n')
    self.assertEqual(test_name, output_name)
    self.assertEqual(test_email, output_email)