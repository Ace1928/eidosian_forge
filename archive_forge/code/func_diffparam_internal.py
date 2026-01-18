from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
def diffparam_internal(self):
    if LooseVersion(self.version) >= LooseVersion('4.0.0'):
        before = self.info.get('internal', False)
        after = self.params['internal']
        return self._diff_update_and_compare('internal', before, after)
    try:
        before = not self.info['plugins'][0]['isgateway']
    except (IndexError, KeyError):
        before = False
    after = self.params['internal']
    return self._diff_update_and_compare('internal', before, after)