from __future__ import absolute_import, division, print_function
import errno
import filecmp
import grp
import os
import os.path
import platform
import pwd
import shutil
import stat
import tempfile
import traceback
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.six import PY3
def split_pre_existing_dir(dirname):
    """
    Return the first pre-existing directory and a list of the new directories that will be created.
    """
    head, tail = os.path.split(dirname)
    b_head = to_bytes(head, errors='surrogate_or_strict')
    if head == '':
        return ('.', [tail])
    if not os.path.exists(b_head):
        if head == '/':
            raise AnsibleModuleError(results={'msg': "The '/' directory doesn't exist on this machine."})
        pre_existing_dir, new_directory_list = split_pre_existing_dir(head)
    else:
        return (head, [tail])
    new_directory_list.append(tail)
    return (pre_existing_dir, new_directory_list)