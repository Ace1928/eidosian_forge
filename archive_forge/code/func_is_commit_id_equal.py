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
def is_commit_id_equal(cls, dest: str, name: Optional[str]) -> bool:
    """Always assume the versions don't match"""
    return False