from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def make_restarted(self):
    """Run actions if desired state is 'restarted'."""
    if self.pod.exists:
        self.pod.restart()
        self.results['actions'].append('restarted %s' % self.pod.name)
        self.results.update({'changed': True})
        self.update_pod_result()
    else:
        self.module.fail_json("Pod %s doesn't exist!" % self.pod.name)