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
def restore_readline() -> None:
    """Restore readline tab completion and history"""
    nonlocal readline_configured
    if not readline_configured:
        return
    if self._completion_supported():
        readline.set_completer(saved_completer)
    if saved_history is not None:
        readline.clear_history()
        for item in saved_history:
            readline.add_history(item)
    readline_configured = False