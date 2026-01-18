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
def set_remote_branch(git_path, module, dest, remote, version, depth):
    """set refs for the remote branch version

    This assumes the branch does not yet exist locally and is therefore also not checked out.
    Can't use git remote set-branches, as it is not available in git 1.7.1 (centos6)
    """
    branchref = '+refs/heads/%s:refs/heads/%s' % (version, version)
    branchref += ' +refs/heads/%s:refs/remotes/%s/%s' % (version, remote, version)
    cmd = '%s fetch --depth=%s %s %s' % (git_path, depth, remote, branchref)
    rc, out, err = module.run_command(cmd, cwd=dest)
    if rc != 0:
        module.fail_json(msg='Failed to fetch branch from remote: %s' % version, stdout=out, stderr=err, rc=rc)