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
@always_prefix_settables.setter
def always_prefix_settables(self, new_value: bool) -> None:
    """
        Set whether CommandSet settable values should always be prefixed.

        :param new_value: True if CommandSet settable values should always be prefixed. False if not.
        :raises ValueError: If a registered CommandSet does not have a defined prefix
        """
    if not self._always_prefix_settables and new_value:
        for cmd_set in self._installed_command_sets:
            if not cmd_set.settable_prefix:
                raise ValueError(f'Cannot force settable prefixes. CommandSet {cmd_set.__class__.__name__} does not have a settable prefix defined.')
    self._always_prefix_settables = new_value