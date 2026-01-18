import sys
import os
import re
import logging
from .errors import DistutilsOptionError
from . import util, dir_util, file_util, archive_util, _modified
from ._log import log
def ensure_dirname(self, option):
    self._ensure_tested_string(option, os.path.isdir, 'directory name', "'%s' does not exist or is not a directory")