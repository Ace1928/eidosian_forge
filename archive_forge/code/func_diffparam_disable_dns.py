from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
def diffparam_disable_dns(self):
    if LooseVersion(self.version) >= LooseVersion('4.0.0'):
        before = not self.info.get('dns_enabled', True)
        after = self.params['disable_dns']
        if self.params['disable_dns'] is None:
            after = before
        return self._diff_update_and_compare('disable_dns', before, after)
    before = after = self.params['disable_dns']
    return self._diff_update_and_compare('disable_dns', before, after)