from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
def diffparam_options(self):
    before = self.info['options'] if 'options' in self.info else {}
    before.pop('uid', None)
    before.pop('gid', None)
    before = ['='.join((k, v)) for k, v in before.items()]
    after = self.params['options']
    before, after = (sorted(list(set(before))), sorted(list(set(after))))
    return self._diff_update_and_compare('options', before, after)