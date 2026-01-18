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
def find_commandset_for_command(self, command_name: str) -> Optional[CommandSet]:
    """
        Finds the CommandSet that registered the command name
        :param command_name: command name to search
        :return: CommandSet that provided the command
        """
    return self._cmd_to_command_sets.get(command_name)