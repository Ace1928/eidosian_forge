from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
def make_present(self):
    """Run actions if desired state is 'started'."""
    if not self.volume.exists:
        self.volume.create()
        self.results['actions'].append('created %s' % self.volume.name)
        self.update_volume_result()
    elif self.recreate or self.volume.different:
        self.volume.recreate()
        self.results['actions'].append('recreated %s' % self.volume.name)
        self.update_volume_result()
    else:
        self.update_volume_result(changed=False)