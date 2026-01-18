from __future__ import annotations
import collections.abc
import sys
import textwrap
import traceback
from functools import singledispatch
from types import TracebackType
from typing import Any, List, Optional
from ._exceptions import BaseExceptionGroup
def exceptiongroup_excepthook(etype: type[BaseException], value: BaseException, tb: TracebackType | None) -> None:
    sys.stderr.write(''.join(traceback.format_exception(etype, value, tb)))