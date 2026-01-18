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
def exclude_by_filter(self, absolute_filename, module_name):
    """
        :return: True if it should be excluded, False if it should be included and None
            if no rule matched the given file.
        """
    for exclude_filter in self._exclude_filters:
        if exclude_filter.is_path:
            if glob_matches_path(absolute_filename, exclude_filter.name):
                return exclude_filter.exclude
        elif exclude_filter.name == module_name or module_name.startswith(exclude_filter.name + '.'):
            return exclude_filter.exclude
    return None