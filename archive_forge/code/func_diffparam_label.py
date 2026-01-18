from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def diffparam_label(self):
    if 'config' in self.info and 'labels' in self.info['config']:
        before = self.info['config'].get('labels') or {}
    else:
        before = self.info['labels'] if 'labels' in self.info else {}
    after = self.params['label']
    if 'podman_systemd_unit' in before:
        after.pop('podman_systemd_unit', None)
        before.pop('podman_systemd_unit', None)
    return self._diff_update_and_compare('label', before, after)