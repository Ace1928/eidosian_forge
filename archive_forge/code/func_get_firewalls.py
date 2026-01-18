from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.firewalls import (
def get_firewalls(self):
    try:
        if self.module.params.get('id') is not None:
            self.hcloud_firewall_info = [self.client.firewalls.get_by_id(self.module.params.get('id'))]
        elif self.module.params.get('name') is not None:
            self.hcloud_firewall_info = [self.client.firewalls.get_by_name(self.module.params.get('name'))]
        elif self.module.params.get('label_selector') is not None:
            self.hcloud_firewall_info = self.client.firewalls.get_all(label_selector=self.module.params.get('label_selector'))
        else:
            self.hcloud_firewall_info = self.client.firewalls.get_all()
    except HCloudException as exception:
        self.fail_json_hcloud(exception)