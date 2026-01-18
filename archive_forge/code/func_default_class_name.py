from __future__ import annotations
import itertools
import os
import re
import typing
from functools import lru_cache
from textwrap import dedent, indent
from the `ggplot()`{.py} call is used. If specified, it overrides the \
from showing in the legend. e.g `show_legend={'color': False}`{.py}, \
def default_class_name(s: str | type | object) -> str:
    """
    Return the qualified name of s

    Only if s does not start with the prefix

    Examples
    --------
    >>> qualified_name('stat_bin')
    'stat_bin'
    >>> qualified_name(stat_bin)
    'stat_bin'
    >>> qualified_name(stat_bin())
    'stat_bin'
    """
    if isinstance(s, str):
        return s
    elif isinstance(s, type):
        s = s.__name__
    else:
        s = s.__class__.__name__
    return s