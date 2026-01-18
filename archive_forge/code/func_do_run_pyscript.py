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
@with_argparser(run_pyscript_parser)
def do_run_pyscript(self, args: argparse.Namespace) -> Optional[bool]:
    """
        Run a Python script file inside the console

        :return: True if running of commands should stop
        """
    self.last_result = False
    args.script_path = os.path.expanduser(args.script_path)
    if not args.script_path.endswith('.py'):
        self.pwarning(f"'{args.script_path}' does not have a .py extension")
        selection = self.select('Yes No', 'Continue to try to run it as a Python script? ')
        if selection != 'Yes':
            return None
    orig_args = sys.argv
    try:
        sys.argv = [args.script_path] + args.script_arguments
        py_return = self._run_python(pyscript=args.script_path)
    finally:
        sys.argv = orig_args
    return py_return