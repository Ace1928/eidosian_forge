from __future__ import annotations
import itertools
import os
import re
import typing
from functools import lru_cache
from textwrap import dedent, indent
from the `ggplot()`{.py} call is used. If specified, it overrides the \
from showing in the legend. e.g `show_legend={'color': False}`{.py}, \
def param_spec(line: str) -> str | None:
    """
    Identify and return parameter

    Parameters
    ----------
    line : str
        A line in the parameter section.

    Returns
    -------
    name : str | None
        Name of the parameter if the line for the parameter
        type specification and None otherwise.

    Examples
    --------
    >>> param_spec('line : str')
    breaks
    >>> param_spec("    A line in the parameter section.")
    """
    m = PARAM_PATTERN.match(line)
    return m.group(1) if m else None