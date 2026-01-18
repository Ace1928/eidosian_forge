from __future__ import absolute_import, division, print_function
import re
import time
import string
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils._text import to_text, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def select_host(self):
    hostsystem = self.cache.get_esx_host(self.params['esxi_hostname'])
    if not hostsystem:
        self.module.fail_json(msg='Failed to find ESX host "%(esxi_hostname)s"' % self.params)
    if hostsystem.runtime.connectionState != 'connected' or hostsystem.runtime.inMaintenanceMode:
        self.module.fail_json(msg='ESXi "%(esxi_hostname)s" is in invalid state or in maintenance mode.' % self.params)
    return hostsystem