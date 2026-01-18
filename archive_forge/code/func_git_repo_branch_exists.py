import shutil
from pathlib import Path
from typing import Tuple
from wasabi import msg
from .commands import run_command
from .filesystem import is_subpath_of, make_tempdir
def git_repo_branch_exists(repo: str, branch: str) -> bool:
    """Uses 'git ls-remote' to check if a repository and branch exists

    repo (str): URL to get repo.
    branch (str): Branch on repo to check.
    RETURNS (bool): True if repo:branch exists.
    """
    get_git_version()
    cmd = f'git ls-remote {repo} {branch}'
    ret = run_command(cmd, capture=True)
    exists = ret.stdout != ''
    return exists