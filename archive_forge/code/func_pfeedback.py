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
def pfeedback(self, msg: Any, *, end: str='\n') -> None:
    """For printing nonessential feedback.  Can be silenced with `quiet`.
        Inclusion in redirected output is controlled by `feedback_to_output`.

        :param msg: object to print
        :param end: string appended after the end of the message, default a newline
        """
    if not self.quiet:
        if self.feedback_to_output:
            self.poutput(msg, end=end)
        else:
            self.perror(msg, end=end, apply_style=False)