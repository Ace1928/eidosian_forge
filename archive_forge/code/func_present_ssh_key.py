from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.ssh_keys import BoundSSHKey
def present_ssh_key(self):
    self._get_ssh_key()
    if self.hcloud_ssh_key is None:
        self._create_ssh_key()
    else:
        self._update_ssh_key()