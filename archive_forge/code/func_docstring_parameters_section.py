from __future__ import annotations
import itertools
import os
import re
import typing
from functools import lru_cache
from textwrap import dedent, indent
from the `ggplot()`{.py} call is used. If specified, it overrides the \
from showing in the legend. e.g `show_legend={'color': False}`{.py}, \
def docstring_parameters_section(obj: Any) -> str:
    """
    Return the parameters section of a docstring
    """
    return docstring_section_lines(obj.__doc__, 'parameters')