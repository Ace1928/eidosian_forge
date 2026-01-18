from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def get_vm_guest_ip(self):
    vm_guest_ip = self.module.params.get('vm_guest_ip')
    default_nic = self.get_vm_default_nic()
    if not vm_guest_ip:
        return default_nic['ipaddress']
    for secondary_ip in default_nic['secondaryip']:
        if vm_guest_ip == secondary_ip['ipaddress']:
            return vm_guest_ip
    self.fail_json(msg="Secondary IP '%s' not assigned to VM" % vm_guest_ip)