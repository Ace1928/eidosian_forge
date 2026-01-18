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
def basic_complete(self, text: str, line: str, begidx: int, endidx: int, match_against: Iterable[str]) -> List[str]:
    """
        Basic tab completion function that matches against a list of strings without considering line contents
        or cursor position. The args required by this function are defined in the header of Python's cmd.py.

        :param text: the string prefix we are attempting to match (all matches must begin with it)
        :param line: the current input line with leading whitespace removed
        :param begidx: the beginning index of the prefix text
        :param endidx: the ending index of the prefix text
        :param match_against: the strings being matched against
        :return: a list of possible tab completions
        """
    return [cur_match for cur_match in match_against if cur_match.startswith(text)]