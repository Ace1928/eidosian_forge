from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING
from .notebook import run_notebook_hook
from .state import curstate
def reset_output(state: State | None=None) -> None:
    """ Clear the default state of all output modes.

    Returns:
        None

    """
    curstate().reset()