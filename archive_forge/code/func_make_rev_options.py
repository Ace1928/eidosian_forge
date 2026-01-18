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
def make_rev_options(cls, rev: Optional[str]=None, extra_args: Optional[CommandArgs]=None) -> RevOptions:
    """
        Return a RevOptions object.

        Args:
          rev: the name of a revision to install.
          extra_args: a list of extra options.
        """
    return RevOptions(cls, rev, extra_args=extra_args)