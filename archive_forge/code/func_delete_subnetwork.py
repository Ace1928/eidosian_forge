from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork, NetworkSubnet
def delete_subnetwork(self):
    self._get_network()
    self._get_subnetwork()
    if self.hcloud_subnetwork is not None and self.hcloud_network is not None:
        if not self.module.check_mode:
            try:
                self.hcloud_network.delete_subnet(self.hcloud_subnetwork).wait_until_finished()
            except HCloudException as exception:
                self.fail_json_hcloud(exception)
        self._mark_as_changed()
    self.hcloud_subnetwork = None