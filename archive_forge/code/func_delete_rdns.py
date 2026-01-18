from __future__ import annotations
import ipaddress
from typing import Any
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.floating_ips import BoundFloatingIP
from ..module_utils.vendor.hcloud.load_balancers import BoundLoadBalancer
from ..module_utils.vendor.hcloud.primary_ips import BoundPrimaryIP
from ..module_utils.vendor.hcloud.servers import BoundServer
def delete_rdns(self):
    self._get_resource()
    self._get_rdns()
    if self.hcloud_rdns is not None:
        if not self.module.check_mode:
            try:
                self.hcloud_resource.change_dns_ptr(ip=self.hcloud_rdns['ip_address'], dns_ptr=None)
            except HCloudException as exception:
                self.fail_json_hcloud(exception)
        self._mark_as_changed()
    self.hcloud_rdns = None