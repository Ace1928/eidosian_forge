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
def switch_version(git_path, module, dest, remote, version, verify_commit, depth, gpg_whitelist):
    cmd = ''
    if version == 'HEAD':
        branch = get_head_branch(git_path, module, dest, remote)
        rc, out, err = module.run_command('%s checkout --force %s' % (git_path, branch), cwd=dest)
        if rc != 0:
            module.fail_json(msg='Failed to checkout branch %s' % branch, stdout=out, stderr=err, rc=rc)
        cmd = '%s reset --hard %s/%s --' % (git_path, remote, branch)
    elif is_remote_branch(git_path, module, dest, remote, version):
        if depth and (not is_local_branch(git_path, module, dest, version)):
            set_remote_branch(git_path, module, dest, remote, version, depth)
        if not is_local_branch(git_path, module, dest, version):
            cmd = '%s checkout --track -b %s %s/%s' % (git_path, version, remote, version)
        else:
            rc, out, err = module.run_command('%s checkout --force %s' % (git_path, version), cwd=dest)
            if rc != 0:
                module.fail_json(msg='Failed to checkout branch %s' % version, stdout=out, stderr=err, rc=rc)
            cmd = '%s reset --hard %s/%s' % (git_path, remote, version)
    else:
        cmd = '%s checkout --force %s' % (git_path, version)
    rc, out1, err1 = module.run_command(cmd, cwd=dest)
    if rc != 0:
        if version != 'HEAD':
            module.fail_json(msg='Failed to checkout %s' % version, stdout=out1, stderr=err1, rc=rc, cmd=cmd)
        else:
            module.fail_json(msg='Failed to checkout branch %s' % branch, stdout=out1, stderr=err1, rc=rc, cmd=cmd)
    if verify_commit:
        verify_commit_sign(git_path, module, dest, version, gpg_whitelist)
    return (rc, out1, err1)