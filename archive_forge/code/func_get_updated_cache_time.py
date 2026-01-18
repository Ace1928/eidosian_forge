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
def get_updated_cache_time():
    """Return the mtime time stamp and the updated cache time.
    Always retrieve the mtime of the apt cache or set the `cache_mtime`
    variable to 0
    :returns: ``tuple``
    """
    cache_mtime = get_cache_mtime()
    mtimestamp = datetime.datetime.fromtimestamp(cache_mtime)
    updated_cache_time = int(time.mktime(mtimestamp.timetuple()))
    return (mtimestamp, updated_cache_time)