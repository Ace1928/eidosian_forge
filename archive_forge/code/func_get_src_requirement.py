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
@classmethod
def get_src_requirement(cls, repo_dir: str, project_name: str) -> str:
    """
        Return the requirement string to use to redownload the files
        currently at the given repository directory.

        Args:
          project_name: the (unescaped) project name.

        The return value has a form similar to the following:

            {repository_url}@{revision}#egg={project_name}
        """
    repo_url = cls.get_remote_url(repo_dir)
    if cls.should_add_vcs_url_prefix(repo_url):
        repo_url = f'{cls.name}+{repo_url}'
    revision = cls.get_requirement_revision(repo_dir)
    subdir = cls.get_subdirectory(repo_dir)
    req = make_vcs_requirement_url(repo_url, revision, project_name, subdir=subdir)
    return req