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
def register_cmdfinalization_hook(self, func: Callable[[plugin.CommandFinalizationData], plugin.CommandFinalizationData]) -> None:
    """Register a hook to be called after a command is completed, whether it completes successfully or not."""
    self._validate_cmdfinalization_callable(func)
    self._cmdfinalization_hooks.append(func)