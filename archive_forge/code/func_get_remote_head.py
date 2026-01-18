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
def get_remote_head(git_path, module, dest, version, remote, bare):
    cloning = False
    cwd = None
    tag = False
    if remote == module.params['repo']:
        cloning = True
    elif remote == 'file://' + os.path.expanduser(module.params['repo']):
        cloning = True
    else:
        cwd = dest
    if version == 'HEAD':
        if cloning:
            cmd = '%s ls-remote %s -h HEAD' % (git_path, remote)
        else:
            head_branch = get_head_branch(git_path, module, dest, remote, bare)
            cmd = '%s ls-remote %s -h refs/heads/%s' % (git_path, remote, head_branch)
    elif is_remote_branch(git_path, module, dest, remote, version):
        cmd = '%s ls-remote %s -h refs/heads/%s' % (git_path, remote, version)
    elif is_remote_tag(git_path, module, dest, remote, version):
        tag = True
        cmd = '%s ls-remote %s -t refs/tags/%s*' % (git_path, remote, version)
    else:
        return version
    rc, out, err = module.run_command(cmd, check_rc=True, cwd=cwd)
    if len(out) < 1:
        module.fail_json(msg='Could not determine remote revision for %s' % version, stdout=out, stderr=err, rc=rc)
    out = to_native(out)
    if tag:
        for tag in out.split('\n'):
            if tag.endswith(version + '^{}'):
                out = tag
                break
            elif tag.endswith(version):
                out = tag
    rev = out.split()[0]
    return rev