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
def enable_category(self, category: str) -> None:
    """
        Enable an entire category of commands

        :param category: the category to enable
        """
    for cmd_name in list(self.disabled_commands):
        func = self.disabled_commands[cmd_name].command_function
        if getattr(func, constants.CMD_ATTR_HELP_CATEGORY, None) == category:
            self.enable_command(cmd_name)