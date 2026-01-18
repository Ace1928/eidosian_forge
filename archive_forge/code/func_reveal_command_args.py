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
def reveal_command_args(args: Union[List[str], CommandArgs]) -> List[str]:
    """
    Return the arguments in their raw, unredacted form.
    """
    return [arg.secret if isinstance(arg, HiddenText) else arg for arg in args]