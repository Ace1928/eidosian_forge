import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def poutput(self, msg: Any='', *, end: str='\n') -> None:
    """Print message to self.stdout and appends a newline by default

        Also handles BrokenPipeError exceptions for when a command's output has
        been piped to another process and that process terminates before the
        cmd2 command is finished executing.

        :param msg: object to print
        :param end: string appended after the end of the message, default a newline
        """
    try:
        ansi.style_aware_write(self.stdout, f'{msg}{end}')
    except BrokenPipeError:
        if self.broken_pipe_warning:
            sys.stderr.write(self.broken_pipe_warning)