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
def allow_style_type(value: str) -> ansi.AllowStyle:
    """Converts a string value into an ansi.AllowStyle"""
    try:
        return ansi.AllowStyle[value.upper()]
    except KeyError:
        raise ValueError(f'must be {ansi.AllowStyle.ALWAYS}, {ansi.AllowStyle.NEVER}, or {ansi.AllowStyle.TERMINAL} (case-insensitive)')