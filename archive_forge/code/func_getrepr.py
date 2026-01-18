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
def getrepr(self, showlocals: bool=False, style: _TracebackStyle='long', abspath: bool=False, tbfilter: Union[bool, Callable[['ExceptionInfo[BaseException]'], Traceback]]=True, funcargs: bool=False, truncate_locals: bool=True, chain: bool=True) -> Union['ReprExceptionInfo', 'ExceptionChainRepr']:
    """Return str()able representation of this exception info.

        :param bool showlocals:
            Show locals per traceback entry.
            Ignored if ``style=="native"``.

        :param str style:
            long|short|line|no|native|value traceback style.

        :param bool abspath:
            If paths should be changed to absolute or left unchanged.

        :param tbfilter:
            A filter for traceback entries.

            * If false, don't hide any entries.
            * If true, hide internal entries and entries that contain a local
              variable ``__tracebackhide__ = True``.
            * If a callable, delegates the filtering to the callable.

            Ignored if ``style`` is ``"native"``.

        :param bool funcargs:
            Show fixtures ("funcargs" for legacy purposes) per traceback entry.

        :param bool truncate_locals:
            With ``showlocals==True``, make sure locals can be safely represented as strings.

        :param bool chain:
            If chained exceptions in Python 3 should be shown.

        .. versionchanged:: 3.9

            Added the ``chain`` parameter.
        """
    if style == 'native':
        return ReprExceptionInfo(reprtraceback=ReprTracebackNative(traceback.format_exception(self.type, self.value, self.traceback[0]._rawentry if self.traceback else None)), reprcrash=self._getreprcrash())
    fmt = FormattedExcinfo(showlocals=showlocals, style=style, abspath=abspath, tbfilter=tbfilter, funcargs=funcargs, truncate_locals=truncate_locals, chain=chain)
    return fmt.repr_excinfo(self)