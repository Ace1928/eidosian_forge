import pydoc
from types import TracebackType
from typing import Optional, Type
from .._typing_compat import Literal
from .. import _internal
def pager(self, output):
    self._repl.pager(output)