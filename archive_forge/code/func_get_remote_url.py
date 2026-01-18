import configparser
import logging
import os
from typing import List, Optional, Tuple
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import HiddenText, display_path
from pip._internal.utils.subprocess import make_command
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs.versioncontrol import (
@classmethod
def get_remote_url(cls, location: str) -> str:
    url = cls.run_command(['showconfig', 'paths.default'], show_stdout=False, stdout_only=True, cwd=location).strip()
    if cls._is_local_repository(url):
        url = path_to_url(url)
    return url.strip()