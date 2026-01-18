import sys
import os
import re
import warnings
from .errors import (
from .spawn import spawn
from .file_util import move_file
from .dir_util import mkpath
from ._modified import newer_group
from .util import split_quoted, execute
from ._log import log
def define_macro(self, name, value=None):
    """Define a preprocessor macro for all compilations driven by this
        compiler object.  The optional parameter 'value' should be a
        string; if it is not supplied, then the macro will be defined
        without an explicit value and the exact outcome depends on the
        compiler used (XXX true? does ANSI say anything about this?)
        """
    i = self._find_macro(name)
    if i is not None:
        del self.macros[i]
    self.macros.append((name, value))