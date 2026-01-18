from __future__ import annotations
import functools
import logging
import os
import subprocess
import sys
import warnings
from datetime import datetime
from typing import TYPE_CHECKING
def raise_deadline_warning() -> None:
    """Raise CI warning after removal deadline in code owner's repo."""

    def _is_in_owner_repo() -> bool:
        """Check if is running in code owner's repo.
            Only generate reliable check when `git` is installed and remote name
            is "origin".
            """
        try:
            result = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], stdout=subprocess.PIPE)
            owner_repo = result.stdout.decode('utf-8').strip().lstrip('https://github.com/').lstrip('git@github.com:').rstrip('.git')
            return owner_repo == os.getenv('GITHUB_REPOSITORY')
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    if _deadline is not None and os.getenv('CI') is not None and (datetime.now() > _deadline) and _is_in_owner_repo():
        raise DeprecationWarning(f'This function should have been removed on {_deadline:%Y-%m-%d}.')