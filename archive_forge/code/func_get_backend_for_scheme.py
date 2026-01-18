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
def get_backend_for_scheme(self, scheme: str) -> Optional['VersionControl']:
    """
        Return a VersionControl object or None.
        """
    for vcs_backend in self._registry.values():
        if scheme in vcs_backend.schemes:
            return vcs_backend
    return None