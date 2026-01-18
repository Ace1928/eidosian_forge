from __future__ import absolute_import, division, print_function
import abc
import os
import stat
import traceback
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
def backup_and_restore(self, sources_and_destinations, *args, **kwargs):
    backups = [(d, self.module.backup_local(d)) for s, d in sources_and_destinations if os.path.exists(d)]
    try:
        f(self, sources_and_destinations, *args, **kwargs)
    except Exception:
        for destination, backup in backups:
            self.module.atomic_move(backup, destination)
        raise
    else:
        for destination, backup in backups:
            self.module.add_cleanup_file(backup)