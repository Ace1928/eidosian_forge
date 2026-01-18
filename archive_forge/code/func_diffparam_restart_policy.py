from __future__ import (absolute_import, division, print_function)
import json  # noqa: F402
import os  # noqa: F402
import shlex  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import normalize_signal
from ansible_collections.containers.podman.plugins.module_utils.podman.common import ARGUMENTS_OPTS_DICT
def diffparam_restart_policy(self):
    before = self.info['hostconfig']['restartpolicy']['name']
    before_max_count = int(self.info['hostconfig']['restartpolicy'].get('maximumretrycount', 0))
    after = self.params['restart_policy'] or ''
    if ':' in after:
        after, after_max_count = after.rsplit(':', 1)
        after_max_count = int(after_max_count)
    else:
        after_max_count = 0
    before = '%s:%i' % (before, before_max_count)
    after = '%s:%i' % (after, after_max_count)
    return self._diff_update_and_compare('restart_policy', before, after)