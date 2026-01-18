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
def get_field_of_deb(m, deb_file, field='Version'):
    cmd_dpkg = m.get_bin_path('dpkg', True)
    cmd = cmd_dpkg + ' --field %s %s' % (deb_file, field)
    rc, stdout, stderr = m.run_command(cmd)
    if rc != 0:
        m.fail_json(msg='%s failed' % cmd, stdout=stdout, stderr=stderr)
    return to_native(stdout).strip('\n')