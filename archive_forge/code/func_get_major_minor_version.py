import functools
import os
import site
import sys
import sysconfig
import typing
from pip._internal.exceptions import InstallationError
from pip._internal.utils import appdirs
from pip._internal.utils.virtualenv import running_under_virtualenv
def get_major_minor_version() -> str:
    """
    Return the major-minor version of the current Python as a string, e.g.
    "3.7" or "3.10".
    """
    return '{}.{}'.format(*sys.version_info)