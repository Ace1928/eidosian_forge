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
def set_libraries(self, libnames):
    """Set the list of libraries to be included in all links driven by
        this compiler object to 'libnames' (a list of strings).  This does
        not affect any standard system libraries that the linker may
        include by default.
        """
    self.libraries = libnames[:]