from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
from ..module_utils.vendor.hcloud.servers import BoundServer
def present_load_balancer_target(self):
    self._get_load_balancer_and_target()
    self._get_load_balancer_target()
    if self.hcloud_load_balancer_target is None:
        self._create_load_balancer_target()