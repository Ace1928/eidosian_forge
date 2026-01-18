from __future__ import annotations
import operator
import sys
import os
import re as _re
import struct
from textwrap import fill, dedent
def filldedent(s, w=70, **kwargs):
    """
    Strips leading and trailing empty lines from a copy of ``s``, then dedents,
    fills and returns it.

    Empty line stripping serves to deal with docstrings like this one that
    start with a newline after the initial triple quote, inserting an empty
    line at the beginning of the string.

    Additional keyword arguments will be passed to ``textwrap.fill()``.

    See Also
    ========
    strlines, rawlines

    """
    return '\n' + fill(dedent(str(s)).strip('\n'), width=w, **kwargs)