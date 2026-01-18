from __future__ import annotations
import itertools
import linecache
import os
import re
import sys
import sysconfig
import traceback
import typing as t
from markupsafe import escape
from ..utils import cached_property
from .console import Console
@cached_property
def all_tracebacks(self) -> list[tuple[str | None, traceback.TracebackException]]:
    out = []
    current = self._te
    while current is not None:
        if current.__cause__ is not None:
            chained_msg = 'The above exception was the direct cause of the following exception'
            chained_exc = current.__cause__
        elif current.__context__ is not None and (not current.__suppress_context__):
            chained_msg = 'During handling of the above exception, another exception occurred'
            chained_exc = current.__context__
        else:
            chained_msg = None
            chained_exc = None
        out.append((chained_msg, current))
        current = chained_exc
    return out