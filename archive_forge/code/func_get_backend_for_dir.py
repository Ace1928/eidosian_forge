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
def get_backend_for_dir(self, location: str) -> Optional['VersionControl']:
    """
        Return a VersionControl object if a repository of that type is found
        at the given directory.
        """
    vcs_backends = {}
    for vcs_backend in self._registry.values():
        repo_path = vcs_backend.get_repository_root(location)
        if not repo_path:
            continue
        logger.debug('Determine that %s uses VCS: %s', location, vcs_backend.name)
        vcs_backends[repo_path] = vcs_backend
    if not vcs_backends:
        return None
    inner_most_repo_path = max(vcs_backends, key=len)
    return vcs_backends[inner_most_repo_path]