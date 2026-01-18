from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork
from ..module_utils.vendor.hcloud.servers import BoundServer, PrivateNet
def present_server_network(self):
    self._get_server_and_network()
    self._get_server_network()
    if self.hcloud_server_network is None:
        self._create_server_network()
    else:
        self._update_server_network()