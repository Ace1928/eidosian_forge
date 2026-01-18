import inspect
import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
import click
import colorama
import ray  # noqa: F401
def print(self, msg: str, *args: Any, _level_str: str='INFO', end: str=None, **kwargs: Any):
    """Prints a message.

        For arguments, see `_format_msg`.
        """
    self._print(_format_msg(msg, *args, **kwargs), _level_str=_level_str, end=end)