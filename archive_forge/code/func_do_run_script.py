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
@with_argparser(run_script_parser)
def do_run_script(self, args: argparse.Namespace) -> Optional[bool]:
    """Run commands in script file that is encoded as either ASCII or UTF-8 text.

        :return: True if running of commands should stop
        """
    self.last_result = False
    expanded_path = os.path.abspath(os.path.expanduser(args.script_path))
    if expanded_path.endswith('.py'):
        self.pwarning(f"'{expanded_path}' appears to be a Python file")
        selection = self.select('Yes No', 'Continue to try to run it as a text script? ')
        if selection != 'Yes':
            return None
    try:
        if os.path.getsize(expanded_path) == 0:
            self.last_result = True
            return None
        if not utils.is_text_file(expanded_path):
            self.perror(f"'{expanded_path}' is not an ASCII or UTF-8 encoded text file")
            return None
        with open(expanded_path, encoding='utf-8') as target:
            script_commands = target.read().splitlines()
    except OSError as ex:
        self.perror(f"Problem accessing script from '{expanded_path}': {ex}")
        return None
    orig_script_dir_count = len(self._script_dir)
    try:
        self._script_dir.append(os.path.dirname(expanded_path))
        if args.transcript:
            self._generate_transcript(script_commands, os.path.expanduser(args.transcript))
        else:
            stop = self.runcmds_plus_hooks(script_commands, stop_on_keyboard_interrupt=True)
            self.last_result = True
            return stop
    finally:
        with self.sigint_protection:
            if orig_script_dir_count != len(self._script_dir):
                self._script_dir.pop()
    return None