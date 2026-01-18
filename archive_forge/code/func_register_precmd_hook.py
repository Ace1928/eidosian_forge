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
def register_precmd_hook(self, func: Callable[[plugin.PrecommandData], plugin.PrecommandData]) -> None:
    """Register a hook to be called before the command function."""
    self._validate_prepostcmd_hook(func, plugin.PrecommandData)
    self._precmd_hooks.append(func)