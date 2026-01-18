import logging
import os
import shutil
import sys
import urllib.parse
from typing import (
from pip._internal.cli.spinners import SpinnerInterface
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import (
from pip._internal.utils.subprocess import (
from pip._internal.utils.urls import get_url_scheme
def find_path_to_project_root_from_repo_root(location: str, repo_root: str) -> Optional[str]:
    """
    Find the the Python project's root by searching up the filesystem from
    `location`. Return the path to project root relative to `repo_root`.
    Return None if the project root is `repo_root`, or cannot be found.
    """
    orig_location = location
    while not is_installable_dir(location):
        last_location = location
        location = os.path.dirname(location)
        if location == last_location:
            logger.warning('Could not find a Python project for directory %s (tried all parent directories)', orig_location)
            return None
    if os.path.samefile(repo_root, location):
        return None
    return os.path.relpath(location, repo_root)