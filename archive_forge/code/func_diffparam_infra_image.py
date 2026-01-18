from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def diffparam_infra_image(self):
    if not self.infra_info:
        return self._diff_update_and_compare('infra_image', '', '')
    before = str(self.infra_info['imagename'])
    after = before
    if self.module_params['infra_image']:
        after = self.params['infra_image']
    before = before.replace(':latest', '')
    after = after.replace(':latest', '')
    before = before.split('/')[-1]
    after = after.split('/')[-1]
    return self._diff_update_and_compare('infra_image', before, after)