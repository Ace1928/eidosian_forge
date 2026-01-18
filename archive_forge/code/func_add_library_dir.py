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
def add_library_dir(self, dir):
    """Add 'dir' to the list of directories that will be searched for
        libraries specified to 'add_library()' and 'set_libraries()'.  The
        linker will be instructed to search for libraries in the order they
        are supplied to 'add_library_dir()' and/or 'set_library_dirs()'.
        """
    self.library_dirs.append(dir)