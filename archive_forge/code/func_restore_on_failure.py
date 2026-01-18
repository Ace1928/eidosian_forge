from __future__ import absolute_import, division, print_function
import abc
import os
import stat
import traceback
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
def restore_on_failure(f):

    def backup_and_restore(module, path, *args, **kwargs):
        backup_file = module.backup_local(path) if os.path.exists(path) else None
        try:
            f(module, path, *args, **kwargs)
        except Exception:
            if backup_file is not None:
                module.atomic_move(backup_file, path)
            raise
        else:
            module.add_cleanup_file(backup_file)
    return backup_and_restore