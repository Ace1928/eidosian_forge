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
def get_head_branch(git_path, module, dest, remote, bare=False):
    """
    Determine what branch HEAD is associated with.  This is partly
    taken from lib/ansible/utils/__init__.py.  It finds the correct
    path to .git/HEAD and reads from that file the branch that HEAD is
    associated with.  In the case of a detached HEAD, this will look
    up the branch in .git/refs/remotes/<remote>/HEAD.
    """
    try:
        repo_path = get_repo_path(dest, bare)
    except (IOError, ValueError) as err:
        module.fail_json(msg='Current repo does not have a valid reference to a separate Git dir or it refers to the invalid path', details=to_text(err))
    headfile = os.path.join(repo_path, 'HEAD')
    if is_not_a_branch(git_path, module, dest):
        headfile = os.path.join(repo_path, 'refs', 'remotes', remote, 'HEAD')
    branch = head_splitter(headfile, remote, module=module, fail_on_error=True)
    return branch