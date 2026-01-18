from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def diffparam_share(self):
    if not self.infra_info:
        return self._diff_update_and_compare('share', '', '')
    if 'sharednamespaces' in self.info:
        before = self.info['sharednamespaces']
    elif 'config' in self.info:
        before = [i.split('shares')[1].lower() for i in self.info['config'] if 'shares' in i]
        before.remove('cgroup')
    else:
        before = []
    if self.params['share'] is not None:
        after = self.params['share'].split(',')
    else:
        after = ['uts', 'ipc', 'net']
        if 'net' not in before:
            after.remove('net')
    if self.params['uidmap'] or self.params['gidmap'] or self.params['userns']:
        after.append('user')
    before, after = (sorted(list(set(before))), sorted(list(set(after))))
    return self._diff_update_and_compare('share', before, after)