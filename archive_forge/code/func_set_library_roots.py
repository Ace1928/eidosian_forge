import fnmatch
import glob
import os.path
import sys
from _pydev_bundle import pydev_log
import pydevd_file_utils
import json
from collections import namedtuple
from _pydev_bundle._pydev_saved_modules import threading
from pydevd_file_utils import normcase
from _pydevd_bundle.pydevd_constants import USER_CODE_BASENAMES_STARTING_WITH, \
from _pydevd_bundle import pydevd_constants
def set_library_roots(self, roots):
    self._library_roots = self._fix_roots(roots)
    pydev_log.debug('LIBRARY_ROOTS %s\n' % roots)