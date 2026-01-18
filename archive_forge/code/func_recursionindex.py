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
def recursionindex(self) -> Optional[int]:
    """Return the index of the frame/TracebackEntry where recursion originates if
        appropriate, None if no recursion occurred."""
    cache: Dict[Tuple[Any, int, int], List[Dict[str, Any]]] = {}
    for i, entry in enumerate(self):
        key = (entry.frame.code.path, id(entry.frame.code.raw), entry.lineno)
        values = cache.setdefault(key, [])
        if values:
            f = entry.frame
            loc = f.f_locals
            for otherloc in values:
                if otherloc == loc:
                    return i
        values.append(entry.frame.f_locals)
    return None