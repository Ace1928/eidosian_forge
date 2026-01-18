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
def get_visible_commands(self) -> List[str]:
    """Return a list of commands that have not been hidden or disabled"""
    return [command for command in self.get_all_commands() if command not in self.hidden_commands and command not in self.disabled_commands]