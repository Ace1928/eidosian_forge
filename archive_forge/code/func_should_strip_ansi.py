import contextlib
import io
import os
import shlex
import shutil
import sys
import tempfile
import typing as t
from types import TracebackType
from . import formatting
from . import termui
from . import utils
from ._compat import _find_binary_reader
def should_strip_ansi(stream: t.Optional[t.IO[t.Any]]=None, color: t.Optional[bool]=None) -> bool:
    if color is None:
        return not default_color
    return not color