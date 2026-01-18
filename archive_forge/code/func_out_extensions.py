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
@property
def out_extensions(self):
    return dict.fromkeys(self.src_extensions, self.obj_extension)