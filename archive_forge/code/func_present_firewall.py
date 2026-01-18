from __future__ import annotations
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.firewalls import (
def present_firewall(self):
    self._get_firewall()
    if self.hcloud_firewall is None:
        self._create_firewall()
    else:
        self._update_firewall()