import argparse
import inspect
import numbers
from collections import (
from typing import (
from .ansi import (
from .constants import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .table_creator import (
def consume_argument(arg_state: _ArgumentState) -> None:
    """Consuming token as an argument"""
    arg_state.count += 1
    consumed_arg_values.setdefault(arg_state.action.dest, [])
    consumed_arg_values[arg_state.action.dest].append(token)