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
def get_repo_path(dest, bare):
    if bare:
        repo_path = dest
    else:
        repo_path = os.path.join(dest, '.git')
    if os.path.isfile(repo_path):
        with open(repo_path, 'r') as gitfile:
            data = gitfile.read()
        ref_prefix, gitdir = data.rstrip().split('gitdir: ', 1)
        if ref_prefix:
            raise ValueError('.git file has invalid git dir reference format')
        if os.path.isabs(gitdir):
            repo_path = gitdir
        else:
            repo_path = os.path.join(dest, gitdir)
        if not os.path.isdir(repo_path):
            raise ValueError('%s is not a directory' % repo_path)
    return repo_path