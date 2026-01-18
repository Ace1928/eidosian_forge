from __future__ import annotations
import itertools
import os
import re
import typing
from functools import lru_cache
from textwrap import dedent, indent
from the `ggplot()`{.py} call is used. If specified, it overrides the \
from showing in the legend. e.g `show_legend={'color': False}`{.py}, \
def append_to_section(s: str, docstring: str, section: str) -> str:
    """
    Append string s to a section in the docstring
    """
    idx = -1
    found = False
    for m in SECTIONS_PATTERN.finditer(docstring):
        if section == m.group('section'):
            found = True
        elif found:
            idx = m.start()
            break
    if found:
        if idx == -1:
            s = f'\n{s}'
        top, bottom = (docstring[:idx], docstring[idx:])
        docstring = f'{top}{s}{bottom}'
    return docstring