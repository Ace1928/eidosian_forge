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
def prompt(self, msg: str, *args, **kwargs):
    """Prompt the user for some text input.

        Args:
            msg: The mesage to display to the user before the prompt.

        Returns:
            The string entered by the user.
        """
    complete_str = cf.underlined(msg)
    rendered_message = _format_msg(complete_str, *args, **kwargs)
    if rendered_message and (not msg.endswith('\n')):
        rendered_message += ' '
    self._print(rendered_message, linefeed=False)
    res = ''
    try:
        ans = sys.stdin.readline()
        ans = ans.lower()
        res = ans.strip()
    except KeyboardInterrupt:
        self.newline()
    return res