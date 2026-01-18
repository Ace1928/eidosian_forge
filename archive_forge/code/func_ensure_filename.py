import sys
import os
import re
import logging
from .errors import DistutilsOptionError
from . import util, dir_util, file_util, archive_util, _modified
from ._log import log
def ensure_filename(self, option):
    """Ensure that 'option' is the name of an existing file."""
    self._ensure_tested_string(option, os.path.isfile, 'filename', "'%s' does not exist or is not a file")