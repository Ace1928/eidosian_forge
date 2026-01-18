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
def mark_installed_manually(m, packages):
    if not packages:
        return
    apt_mark_cmd_path = m.get_bin_path('apt-mark')
    if apt_mark_cmd_path is None:
        m.warn('Could not find apt-mark binary, not marking package(s) as manually installed.')
        return
    cmd = '%s manual %s' % (apt_mark_cmd_path, ' '.join(packages))
    rc, out, err = m.run_command(cmd)
    if APT_MARK_INVALID_OP in err or APT_MARK_INVALID_OP_DEB6 in err:
        cmd = '%s unmarkauto %s' % (apt_mark_cmd_path, ' '.join(packages))
        rc, out, err = m.run_command(cmd)
    if rc != 0:
        m.fail_json(msg="'%s' failed: %s" % (cmd, err), stdout=out, stderr=err, rc=rc)