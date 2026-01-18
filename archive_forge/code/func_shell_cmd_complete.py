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
def shell_cmd_complete(self, text: str, line: str, begidx: int, endidx: int, *, complete_blank: bool=False) -> List[str]:
    """Performs completion of executables either in a user's path or a given path

        :param text: the string prefix we are attempting to match (all matches must begin with it)
        :param line: the current input line with leading whitespace removed
        :param begidx: the beginning index of the prefix text
        :param endidx: the ending index of the prefix text
        :param complete_blank: If True, then a blank will complete all shell commands in a user's path. If False, then
                               no completion is performed. Defaults to False to match Bash shell behavior.
        :return: a list of possible tab completions
        """
    if not complete_blank and (not text):
        return []
    if not text.startswith('~') and os.path.sep not in text:
        return utils.get_exes_in_path(text)
    else:
        return self.path_complete(text, line, begidx, endidx, path_filter=lambda path: os.path.isdir(path) or os.access(path, os.X_OK))