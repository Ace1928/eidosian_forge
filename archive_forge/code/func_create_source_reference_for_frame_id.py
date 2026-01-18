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
def create_source_reference_for_frame_id(frame_id, original_filename):
    source_reference = _next_source_reference()
    pydev_log.debug('Created frame id source reference: %s for frame id: %s (%s)', source_reference, frame_id, original_filename)
    _source_reference_to_frame_id[source_reference] = frame_id
    return source_reference