import os
import re
import sys
import typing as t
from functools import update_wrapper
from types import ModuleType
from types import TracebackType
from ._compat import _default_text_stderr
from ._compat import _default_text_stdout
from ._compat import _find_binary_writer
from ._compat import auto_wrap_for_ansi
from ._compat import binary_streams
from ._compat import open_stream
from ._compat import should_strip_ansi
from ._compat import strip_ansi
from ._compat import text_streams
from ._compat import WIN
from .globals import resolve_color_default
def close_intelligently(self) -> None:
    """This function only closes the file if it was opened by the lazy
        file wrapper.  For instance this will never close stdin.
        """
    if self.should_close:
        self.close()