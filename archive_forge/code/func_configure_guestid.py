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
def configure_guestid(self, vm_obj, vm_creation=False):
    if self.params['template']:
        return
    if vm_creation and self.params['guest_id'] is None:
        self.module.fail_json(msg='guest_id attribute is mandatory for VM creation')
    if self.params['guest_id'] and (vm_obj is None or self.params['guest_id'].lower() != vm_obj.summary.config.guestId.lower()):
        self.change_detected = True
        self.configspec.guestId = self.params['guest_id']