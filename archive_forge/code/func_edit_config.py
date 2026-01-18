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
def edit_config(self):
    if not self.config.config_path.is_file():
        if self.interact.confirm(_('Config file does not exist - create new from default? (y/N)')):
            try:
                default_config = pkgutil.get_data('bpython', 'sample-config')
                default_config = default_config.decode('ascii')
                containing_dir = self.config.config_path.parent
                if not containing_dir.exists():
                    containing_dir.mkdir(parents=True)
                with open(self.config.config_path, 'w') as f:
                    f.write(default_config)
            except OSError as e:
                self.interact.notify(_("Error writing file '%s': %s") % (self.config.config_path, e))
                return False
        else:
            return False
    try:
        if self.open_in_external_editor(self.config.config_path):
            self.interact.notify(_('bpython config file edited. Restart bpython for changes to take effect.'))
    except OSError as e:
        self.interact.notify(_('Error editing config file: %s') % e)