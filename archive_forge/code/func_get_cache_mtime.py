from __future__ import absolute_import, division, print_function
import warnings
import datetime
import fnmatch
import locale as locale_module
import os
import random
import re
import shutil
import sys
import tempfile
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.six import PY3, string_types
from ansible.module_utils.urls import fetch_file
def get_cache_mtime():
    """Return mtime of a valid apt cache file.
    Stat the apt cache file and if no cache file is found return 0
    :returns: ``int``
    """
    cache_time = 0
    if os.path.exists(APT_UPDATE_SUCCESS_STAMP_PATH):
        cache_time = os.stat(APT_UPDATE_SUCCESS_STAMP_PATH).st_mtime
    elif os.path.exists(APT_LISTS_PATH):
        cache_time = os.stat(APT_LISTS_PATH).st_mtime
    return cache_time