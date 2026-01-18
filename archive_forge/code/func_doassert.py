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
def doassert(self, val: bool, msg: str, *args: Any, **kwargs: Any):
    """Handle assertion without throwing a scary exception.

        Args:
            val: Value to check.

        For other arguments, see `_format_msg`.
        """
    if not val:
        exc = None
        if not self.pretty:
            exc = AssertionError()
        self.abort(msg, *args, exc=exc, **kwargs)