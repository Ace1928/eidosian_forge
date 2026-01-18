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
def register_postparsing_hook(self, func: Callable[[plugin.PostparsingData], plugin.PostparsingData]) -> None:
    """Register a function to be called after parsing user input but before running the command"""
    self._validate_postparsing_callable(func)
    self._postparsing_hooks.append(func)