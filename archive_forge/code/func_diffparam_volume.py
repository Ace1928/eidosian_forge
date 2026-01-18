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
def diffparam_volume(self):

    def clean_volume(x):
        """Remove trailing and double slashes from volumes."""
        if not x.rstrip('/'):
            return '/'
        return x.replace('//', '/').rstrip('/')
    before = self.info['mounts']
    before_local_vols = []
    if before:
        volumes = []
        local_vols = []
        for m in before:
            if m['type'] != 'volume':
                volumes.append([clean_volume(m['source']), clean_volume(m['destination'])])
            elif m['type'] == 'volume':
                local_vols.append([m['name'], clean_volume(m['destination'])])
        before = [':'.join(v) for v in volumes]
        before_local_vols = [':'.join(v) for v in local_vols]
    if self.params['volume'] is not None:
        after = [':'.join([clean_volume(i) for i in v.split(':')[:2]]) for v in self.params['volume']]
    else:
        after = []
    if before_local_vols:
        after = list(set(after).difference(before_local_vols))
    before, after = (sorted(list(set(before))), sorted(list(set(after))))
    return self._diff_update_and_compare('volume', before, after)