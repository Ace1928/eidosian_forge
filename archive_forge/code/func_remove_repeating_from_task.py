from __future__ import annotations
import io
import re
from functools import partial
from pprint import pformat
from re import Match
from textwrap import fill
from typing import Any, Callable, Pattern
def remove_repeating_from_task(task_name: str, s: str) -> str:
    """Given task name, remove repeating module names.

    Example:
        >>> remove_repeating_from_task(
        ...     'tasks.add',
        ...     'tasks.add(2, 2), tasks.mul(3), tasks.div(4)')
        'tasks.add(2, 2), mul(3), div(4)'
    """
    module = str(task_name).rpartition('.')[0] + '.'
    return remove_repeating(module, s)