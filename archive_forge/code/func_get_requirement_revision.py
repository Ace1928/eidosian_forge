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
def get_requirement_revision(cls, location: str) -> str:
    """
        Return the changeset identification hash, as a 40-character
        hexadecimal string
        """
    current_rev_hash = cls.run_command(['parents', '--template={node}'], show_stdout=False, stdout_only=True, cwd=location).strip()
    return current_rev_hash