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
def complete_set_value(self, text: str, line: str, begidx: int, endidx: int, arg_tokens: Dict[str, List[str]]) -> List[str]:
    """Completes the value argument of set"""
    param = arg_tokens['param'][0]
    try:
        settable = self.settables[param]
    except KeyError:
        raise CompletionError(param + ' is not a settable parameter')
    settable_parser = argparse_custom.DEFAULT_ARGUMENT_PARSER(parents=[Cmd.set_parser_parent])
    arg_name = 'value'
    settable_parser.add_argument(arg_name, metavar=arg_name, help=settable.description, choices=settable.choices, choices_provider=settable.choices_provider, completer=settable.completer)
    completer = argparse_completer.DEFAULT_AP_COMPLETER(settable_parser, self)
    _, raw_tokens = self.tokens_for_completion(line, begidx, endidx)
    return completer.complete(text, line, begidx, endidx, raw_tokens[1:])