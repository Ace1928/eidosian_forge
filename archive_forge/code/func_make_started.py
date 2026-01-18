from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def make_started(self):
    """Run actions if desired state is 'started'."""
    changed = self._create_or_recreate_pod()
    if not changed and self.pod.running:
        self.update_pod_result(changed=changed)
        return
    self.pod.start()
    self.results['actions'].append('started %s' % self.pod.name)
    self.update_pod_result()