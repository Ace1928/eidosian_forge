from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import DebugInfoHolder, IS_WINDOWS, IS_JYTHON, \
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydevd_bundle.pydevd_comm_constants import file_system_encoding, filesystem_encoding_is_utf8
from _pydev_bundle.pydev_log import error_once
import json
import os.path
import sys
import itertools
import ntpath
from functools import partial
def canonical_normalized_path(filename):
    """
    This returns a filename that is canonical and it's meant to be used internally
    to store information on breakpoints and see if there's any hit on it.

    Note that this version is only internal as it may not match the case and
    may have symlinks resolved (and thus may not match what the user expects
    in the editor).
    """
    return get_abs_path_real_path_and_base_from_file(filename)[1]