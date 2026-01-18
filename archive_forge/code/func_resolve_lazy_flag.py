import os
import stat
import sys
import typing as t
from datetime import datetime
from gettext import gettext as _
from gettext import ngettext
from ._compat import _get_argv_encoding
from ._compat import open_stream
from .exceptions import BadParameter
from .utils import format_filename
from .utils import LazyFile
from .utils import safecall
def resolve_lazy_flag(self, value: 't.Union[str, os.PathLike[str]]') -> bool:
    if self.lazy is not None:
        return self.lazy
    if os.fspath(value) == '-':
        return False
    elif 'w' in self.mode:
        return True
    return False