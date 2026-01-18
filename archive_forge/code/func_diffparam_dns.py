from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def diffparam_dns(self):
    if not self.infra_info:
        return self._diff_update_and_compare('dns', '', '')
    before = self.infra_info['hostconfig']['dns'] or []
    after = self.params['dns']
    before, after = (sorted(list(set(before))), sorted(list(set(after))))
    return self._diff_update_and_compare('dns', before, after)