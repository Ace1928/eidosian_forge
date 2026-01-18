from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def diffparam_infra(self):
    if 'state' in self.info and 'infracontainerid' in self.info['state']:
        before = self.info['state']['infracontainerid'] != ''
    else:
        before = 'infracontainerid' in self.info
    after = self.params['infra']
    return self._diff_update_and_compare('infra', before, after)