from __future__ import annotations
import contextlib
import inspect
import os
import re
from typing import TYPE_CHECKING
import warnings
@contextlib.contextmanager
def rewrite_exception(old_name: str, new_name: str) -> Generator[None, None, None]:
    """
    Rewrite the message of an exception.
    """
    try:
        yield
    except Exception as err:
        if not err.args:
            raise
        msg = str(err.args[0])
        msg = msg.replace(old_name, new_name)
        args: tuple[str, ...] = (msg,)
        if len(err.args) > 1:
            args = args + err.args[1:]
        err.args = args
        raise