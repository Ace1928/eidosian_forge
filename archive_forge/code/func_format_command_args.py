import logging
import os
import shlex
import subprocess
from typing import (
from pip._vendor.rich.markup import escape
from pip._internal.cli.spinners import SpinnerInterface, open_spinner
from pip._internal.exceptions import InstallationSubprocessError
from pip._internal.utils.logging import VERBOSE, subprocess_logger
from pip._internal.utils.misc import HiddenText
def format_command_args(args: Union[List[str], CommandArgs]) -> str:
    """
    Format command arguments for display.
    """
    return ' '.join((shlex.quote(str(arg)) if isinstance(arg, HiddenText) else shlex.quote(arg) for arg in args))