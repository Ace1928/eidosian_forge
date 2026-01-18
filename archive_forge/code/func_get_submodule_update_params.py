from __future__ import absolute_import, division, print_function
import filecmp
import os
import re
import shlex
import stat
import sys
import shutil
import tempfile
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.six import b, string_types
def get_submodule_update_params(module, git_path, cwd):
    params = []
    cmd = '%s submodule update --help' % git_path
    rc, stdout, stderr = module.run_command(cmd, cwd=cwd)
    lines = stderr.split('\n')
    update_line = None
    for line in lines:
        if 'git submodule [--quiet] update ' in line:
            update_line = line
    if update_line:
        update_line = update_line.replace('[', '')
        update_line = update_line.replace(']', '')
        update_line = update_line.replace('|', ' ')
        parts = shlex.split(update_line)
        for part in parts:
            if part.startswith('--'):
                part = part.replace('--', '')
                params.append(part)
    return params