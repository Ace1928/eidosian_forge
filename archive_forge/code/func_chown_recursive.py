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
def chown_recursive(path, module):
    changed = False
    owner = module.params['owner']
    group = module.params['group']
    if owner is not None:
        if not module.check_mode:
            for dirpath, dirnames, filenames in os.walk(path):
                owner_changed = module.set_owner_if_different(dirpath, owner, False)
                if owner_changed is True:
                    changed = owner_changed
                for dir in [os.path.join(dirpath, d) for d in dirnames]:
                    owner_changed = module.set_owner_if_different(dir, owner, False)
                    if owner_changed is True:
                        changed = owner_changed
                for file in [os.path.join(dirpath, f) for f in filenames]:
                    owner_changed = module.set_owner_if_different(file, owner, False)
                    if owner_changed is True:
                        changed = owner_changed
        else:
            uid = pwd.getpwnam(owner).pw_uid
            for dirpath, dirnames, filenames in os.walk(path):
                owner_changed = os.stat(dirpath).st_uid != uid
                if owner_changed is True:
                    changed = owner_changed
                for dir in [os.path.join(dirpath, d) for d in dirnames]:
                    owner_changed = os.stat(dir).st_uid != uid
                    if owner_changed is True:
                        changed = owner_changed
                for file in [os.path.join(dirpath, f) for f in filenames]:
                    owner_changed = os.stat(file).st_uid != uid
                    if owner_changed is True:
                        changed = owner_changed
    if group is not None:
        if not module.check_mode:
            for dirpath, dirnames, filenames in os.walk(path):
                group_changed = module.set_group_if_different(dirpath, group, False)
                if group_changed is True:
                    changed = group_changed
                for dir in [os.path.join(dirpath, d) for d in dirnames]:
                    group_changed = module.set_group_if_different(dir, group, False)
                    if group_changed is True:
                        changed = group_changed
                for file in [os.path.join(dirpath, f) for f in filenames]:
                    group_changed = module.set_group_if_different(file, group, False)
                    if group_changed is True:
                        changed = group_changed
        else:
            gid = grp.getgrnam(group).gr_gid
            for dirpath, dirnames, filenames in os.walk(path):
                group_changed = os.stat(dirpath).st_gid != gid
                if group_changed is True:
                    changed = group_changed
                for dir in [os.path.join(dirpath, d) for d in dirnames]:
                    group_changed = os.stat(dir).st_gid != gid
                    if group_changed is True:
                        changed = group_changed
                for file in [os.path.join(dirpath, f) for f in filenames]:
                    group_changed = os.stat(file).st_gid != gid
                    if group_changed is True:
                        changed = group_changed
    return changed