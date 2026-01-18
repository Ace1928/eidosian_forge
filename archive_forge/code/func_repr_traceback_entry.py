import ast
import dataclasses
import inspect
from inspect import CO_VARARGS
from inspect import CO_VARKEYWORDS
from io import StringIO
import os
from pathlib import Path
import re
import sys
import traceback
from traceback import format_exception_only
from types import CodeType
from types import FrameType
from types import TracebackType
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Final
from typing import final
from typing import Generic
from typing import Iterable
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import SupportsIndex
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import pluggy
import _pytest
from _pytest._code.source import findsource
from _pytest._code.source import getrawcode
from _pytest._code.source import getstatementrange_ast
from _pytest._code.source import Source
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import safeformat
from _pytest._io.saferepr import saferepr
from _pytest.compat import get_real_func
from _pytest.deprecated import check_ispytest
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
def repr_traceback_entry(self, entry: Optional[TracebackEntry], excinfo: Optional[ExceptionInfo[BaseException]]=None) -> 'ReprEntry':
    lines: List[str] = []
    style = entry._repr_style if entry is not None and entry._repr_style is not None else self.style
    if style in ('short', 'long') and entry is not None:
        source = self._getentrysource(entry)
        if source is None:
            source = Source('???')
            line_index = 0
        else:
            line_index = entry.lineno - entry.getfirstlinesource()
        short = style == 'short'
        reprargs = self.repr_args(entry) if not short else None
        s = self.get_source(source, line_index, excinfo, short=short)
        lines.extend(s)
        if short:
            message = 'in %s' % entry.name
        else:
            message = excinfo and excinfo.typename or ''
        entry_path = entry.path
        path = self._makepath(entry_path)
        reprfileloc = ReprFileLocation(path, entry.lineno + 1, message)
        localsrepr = self.repr_locals(entry.locals)
        return ReprEntry(lines, reprargs, localsrepr, reprfileloc, style)
    elif style == 'value':
        if excinfo:
            lines.extend(str(excinfo.value).split('\n'))
        return ReprEntry(lines, None, None, None, style)
    else:
        if excinfo:
            lines.extend(self.get_exconly(excinfo, indent=4))
        return ReprEntry(lines, None, None, None, style)