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
def copy_diff_files(src, dest, module):
    """Copy files that are different between `src` directory and `dest` directory."""
    changed = False
    owner = module.params['owner']
    group = module.params['group']
    local_follow = module.params['local_follow']
    diff_files = filecmp.dircmp(src, dest).diff_files
    if len(diff_files):
        changed = True
    if not module.check_mode:
        for item in diff_files:
            src_item_path = os.path.join(src, item)
            dest_item_path = os.path.join(dest, item)
            b_src_item_path = to_bytes(src_item_path, errors='surrogate_or_strict')
            b_dest_item_path = to_bytes(dest_item_path, errors='surrogate_or_strict')
            if os.path.islink(b_src_item_path) and local_follow is False:
                linkto = os.readlink(b_src_item_path)
                os.symlink(linkto, b_dest_item_path)
            else:
                shutil.copyfile(b_src_item_path, b_dest_item_path)
                shutil.copymode(b_src_item_path, b_dest_item_path)
            if owner is not None:
                module.set_owner_if_different(b_dest_item_path, owner, False)
            if group is not None:
                module.set_group_if_different(b_dest_item_path, group, False)
            changed = True
    return changed