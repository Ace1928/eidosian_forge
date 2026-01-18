from __future__ import annotations
import io
import re
from functools import partial
from pprint import pformat
from re import Match
from textwrap import fill
from typing import Any, Callable, Pattern
def remove_repeating(substr: str, s: str) -> str:
    """Remove repeating module names from string.

    Arguments:
        task_name (str): Task name (full path including module),
            to use as the basis for removing module names.
        s (str): The string we want to work on.

    Example:

        >>> _shorten_names(
        ...    'x.tasks.add',
        ...    'x.tasks.add(2, 2) | x.tasks.add(4) | x.tasks.mul(8)',
        ... )
        'x.tasks.add(2, 2) | add(4) | mul(8)'
    """
    index = s.find(substr)
    if index >= 0:
        return ''.join([s[:index + len(substr)], s[index + len(substr):].replace(substr, '')])
    return s