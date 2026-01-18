from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.floating_ips import BoundFloatingIP
def present_floating_ip(self):
    self._get_floating_ip()
    if self.hcloud_floating_ip is None:
        self._create_floating_ip()
    else:
        self._update_floating_ip()