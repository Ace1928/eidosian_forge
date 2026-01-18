from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
def diffparam_gateway(self):
    if LooseVersion(self.version) >= LooseVersion('4.0.0'):
        return self._diff_update_and_compare('gateway', '', '')
    try:
        before = self.info['plugins'][0]['ipam']['ranges'][0][0]['gateway']
    except (IndexError, KeyError):
        before = ''
    after = before
    if self.params['gateway'] is not None:
        after = self.params['gateway']
    return self._diff_update_and_compare('gateway', before, after)