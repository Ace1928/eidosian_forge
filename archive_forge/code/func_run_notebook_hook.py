from __future__ import annotations
import logging # isort:skip
import json
import os
import urllib
from typing import (
from uuid import uuid4
from ..core.types import ID
from ..util.serialization import make_id
from ..util.warnings import warn
from .state import curstate
def run_notebook_hook(notebook_type: NotebookType, action: Literal['load', 'doc', 'app'], *args: Any, **kwargs: Any) -> Any:
    """ Run an installed notebook hook with supplied arguments.

    Args:
        noteboook_type (str) :
            Name of an existing installed notebook hook

        actions (str) :
            Name of the hook action to execute, ``'doc'`` or ``'app'``

    All other arguments and keyword arguments are passed to the hook action
    exactly as supplied.

    Returns:
        Result of the hook action, as-is

    Raises:
        RuntimeError
            If the hook or specific action is not installed

    """
    if notebook_type not in _HOOKS:
        raise RuntimeError(f'no display hook installed for notebook type {notebook_type!r}')
    if _HOOKS[notebook_type][action] is None:
        raise RuntimeError(f'notebook hook for {notebook_type!r} did not install {action!r} action')
    return _HOOKS[notebook_type][action](*args, **kwargs)