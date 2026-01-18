import re
import subprocess
from pathlib import Path
from typing import Optional
import typer
from wasabi import msg
from .. import about
from ..util import ensure_path, get_git_version, git_checkout, git_repo_branch_exists
from .main import COMMAND, PROJECT_FILE, Arg, Opt, _get_parent_command, app
def check_clone(name: str, dest: Path, repo: str) -> None:
    """Check and validate that the destination path can be used to clone. Will
    check that Git is available and that the destination path is suitable.

    name (str): Name of the directory to clone from the repo.
    dest (Path): Local destination of cloned directory.
    repo (str): URL of the repo to clone from.
    """
    git_err = f"Cloning Weasel project templates requires Git and the 'git' command. To clone a project without Git, copy the files from the '{name}' directory in the {repo} to {dest} manually."
    get_git_version(error=git_err)
    if not dest:
        msg.fail(f'Not a valid directory to clone project: {dest}', exits=1)
    if dest.exists():
        msg.fail(f"Can't clone project, directory already exists: {dest}", exits=1)
    if not dest.parent.exists():
        msg.fail(f"Can't clone project, parent directory doesn't exist: {dest.parent}. Create the necessary folder(s) first before continuing.", exits=1)