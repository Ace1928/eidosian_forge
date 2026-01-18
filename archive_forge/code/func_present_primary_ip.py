from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.primary_ips import BoundPrimaryIP
def present_primary_ip(self):
    self._get_primary_ip()
    if self.hcloud_primary_ip is None:
        self._create_primary_ip()
    else:
        self._update_primary_ip()