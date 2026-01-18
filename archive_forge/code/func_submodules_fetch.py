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
def submodules_fetch(git_path, module, remote, track_submodules, dest):
    changed = False
    if not os.path.exists(os.path.join(dest, '.gitmodules')):
        return changed
    gitmodules_file = open(os.path.join(dest, '.gitmodules'), 'r')
    for line in gitmodules_file:
        if not changed and line.strip().startswith('path'):
            path = line.split('=', 1)[1].strip()
            if not os.path.exists(os.path.join(dest, path, '.git')):
                changed = True
    if not changed:
        begin = get_submodule_versions(git_path, module, dest)
        cmd = [git_path, 'submodule', 'foreach', git_path, 'fetch']
        rc, out, err = module.run_command(cmd, check_rc=True, cwd=dest)
        if rc != 0:
            module.fail_json(msg='Failed to fetch submodules: %s' % out + err)
        if track_submodules:
            version = 'master'
            after = get_submodule_versions(git_path, module, dest, '%s/%s' % (remote, version))
            if begin != after:
                changed = True
        else:
            cmd = [git_path, 'submodule', 'status']
            rc, out, err = module.run_command(cmd, check_rc=True, cwd=dest)
            if rc != 0:
                module.fail_json(msg='Failed to retrieve submodule status: %s' % out + err)
            for line in out.splitlines():
                if line[0] != ' ':
                    changed = True
                    break
    return changed