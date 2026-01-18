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
def is_remote_branch(git_path, module, dest, remote, version):
    cmd = '%s ls-remote %s -h refs/heads/%s' % (git_path, remote, version)
    rc, out, err = module.run_command(cmd, check_rc=True, cwd=dest)
    if to_native(version, errors='surrogate_or_strict') in out:
        return True
    else:
        return False