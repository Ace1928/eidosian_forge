from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def make_paused(self):
    """Run actions if desired state is 'paused'."""
    changed = self._create_or_recreate_pod()
    if self.pod.paused:
        self.update_pod_result(changed=changed)
        return
    self.pod.pause()
    self.results['actions'].append('paused %s' % self.pod.name)
    self.update_pod_result()