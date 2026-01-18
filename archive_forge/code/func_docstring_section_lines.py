from __future__ import annotations
import itertools
import os
import re
import typing
from functools import lru_cache
from textwrap import dedent, indent
from the `ggplot()`{.py} call is used. If specified, it overrides the \
from showing in the legend. e.g `show_legend={'color': False}`{.py}, \
@lru_cache(maxsize=256)
def docstring_section_lines(docstring: str, section_name: str) -> str:
    """
    Return a section of a numpydoc string

    Parameters
    ----------
    docstring :
        Docstring
    section_name :
        Name of section to return

    Returns
    -------
    :
        Section minus the header
    """
    lines = []
    inside_section = False
    underline = '-' * len(section_name)
    expect_underline = False
    for line in docstring.splitlines():
        _line = line.strip().lower()
        if expect_underline:
            expect_underline = False
            if _line == underline:
                inside_section = True
                continue
        if _line == section_name:
            expect_underline = True
        elif _line in DOCSTRING_SECTIONS:
            break
        elif inside_section:
            lines.append(line)
    return '\n'.join(lines)