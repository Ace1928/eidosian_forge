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
def complete_help_subcommands(self, text: str, line: str, begidx: int, endidx: int, arg_tokens: Dict[str, List[str]]) -> List[str]:
    """Completes the subcommands argument of help"""
    command = arg_tokens['command'][0]
    if not command:
        return []
    func = self.cmd_func(command)
    argparser = getattr(func, constants.CMD_ATTR_ARGPARSER, None)
    if func is None or argparser is None:
        return []
    completer = argparse_completer.DEFAULT_AP_COMPLETER(argparser, self)
    return completer.complete_subcommand_help(text, line, begidx, endidx, arg_tokens['subcommands'])