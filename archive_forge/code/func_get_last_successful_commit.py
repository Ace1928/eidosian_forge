from __future__ import annotations
import os
import tempfile
import uuid
import typing as t
import urllib.parse
from ..encoding import (
from ..config import (
from ..git import (
from ..http import (
from ..util import (
from . import (
def get_last_successful_commit(self, commits: set[str]) -> t.Optional[str]:
    """Return the last successful commit from git history that is found in the given commit list, or None."""
    commit_history = self.git.get_rev_list(max_count=100)
    ordered_successful_commits = [commit for commit in commit_history if commit in commits]
    last_successful_commit = ordered_successful_commits[0] if ordered_successful_commits else None
    return last_successful_commit