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
def should_add_vcs_url_prefix(cls, remote_url: str) -> bool:
    """
        Return whether the vcs prefix (e.g. "git+") should be added to a
        repository's remote url when used in a requirement.
        """
    return not remote_url.lower().startswith(f'{cls.name}:')