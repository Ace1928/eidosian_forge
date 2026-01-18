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
@staticmethod
def make_rev_args(username: Optional[str], password: Optional[HiddenText]) -> CommandArgs:
    """
        Return the RevOptions "extra arguments" to use in obtain().
        """
    return []