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
def parse_diff(output):
    diff = to_native(output).splitlines()
    try:
        diff_start = diff.index('Resolving dependencies...')
    except ValueError:
        try:
            diff_start = diff.index('Reading state information...')
        except ValueError:
            diff_start = -1
    try:
        diff_end = next((i for i, item in enumerate(diff) if re.match('[0-9]+ (packages )?upgraded', item)))
    except StopIteration:
        diff_end = len(diff)
    diff_start += 1
    diff_end += 1
    return {'prepared': '\n'.join(diff[diff_start:diff_end])}