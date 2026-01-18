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
def make_vcs_requirement_url(repo_url: str, rev: str, project_name: str, subdir: Optional[str]=None) -> str:
    """
    Return the URL for a VCS requirement.

    Args:
      repo_url: the remote VCS url, with any needed VCS prefix (e.g. "git+").
      project_name: the (unescaped) project name.
    """
    egg_project_name = project_name.replace('-', '_')
    req = f'{repo_url}@{rev}#egg={egg_project_name}'
    if subdir:
        req += f'&subdirectory={subdir}'
    return req