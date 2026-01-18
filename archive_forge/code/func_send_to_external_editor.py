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
def send_to_external_editor(self, text: str) -> str:
    """Returns modified text from an editor, or the original text if editor
        exited with non-zero"""
    encoding = getpreferredencoding()
    editor_args = shlex.split(self.config.editor)
    with tempfile.NamedTemporaryFile(suffix='.py') as temp:
        temp.write(text.encode(encoding))
        temp.flush()
        args = editor_args + [temp.name]
        if subprocess.call(args) == 0:
            with open(temp.name) as f:
                return f.read()
        else:
            return text