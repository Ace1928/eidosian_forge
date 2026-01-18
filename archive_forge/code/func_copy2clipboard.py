import abc
import code
import inspect
import os
import pkgutil
import pydoc
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from abc import abstractmethod
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path
from types import ModuleType, TracebackType
from typing import (
from ._typing_compat import Literal
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from . import autocomplete, inspection, simpleeval
from .config import getpreferredencoding, Config
from .formatter import Parenthesis
from .history import History
from .lazyre import LazyReCompile
from .paste import PasteHelper, PastePinnwand, PasteFailed
from .patch_linecache import filename_for_console_input
from .translations import _, ngettext
from .importcompletion import ModuleGatherer
def copy2clipboard(self) -> None:
    """Copy current content to clipboard."""
    if not have_pyperclip:
        self.interact.notify(_('No clipboard available.'))
        return
    content = self.get_session_formatted_for_file()
    try:
        pyperclip.copy(content)
    except pyperclip.PyperclipException:
        self.interact.notify(_('Could not copy to clipboard.'))
    else:
        self.interact.notify(_('Copied content to clipboard.'))