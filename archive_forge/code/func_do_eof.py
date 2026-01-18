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
@with_argparser(eof_parser)
def do_eof(self, _: argparse.Namespace) -> Optional[bool]:
    """
        Called when Ctrl-D is pressed and calls quit with no arguments.
        This can be overridden if quit should be called differently.
        """
    self.poutput()
    return self.do_quit('')