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
def pastebin(self, s=None) -> Optional[str]:
    """Upload to a pastebin and display the URL in the status bar."""
    if s is None:
        s = self.getstdout()
    if self.config.pastebin_confirm and (not self.interact.confirm(_('Pastebin buffer? (y/N) '))):
        self.interact.notify(_('Pastebin aborted.'))
        return None
    else:
        return self.do_pastebin(s)