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
def customize_customvalues(self, vm_obj):
    if not self.params['customvalues']:
        return
    if not self.is_vcenter():
        self.module.warn('Currently connected to ESXi. customvalues are a vCenter feature, this parameter will be ignored.')
        return
    facts = self.gather_facts(vm_obj)
    for kv in self.params['customvalues']:
        if 'key' not in kv or 'value' not in kv:
            self.module.exit_json(msg="customvalues items required both 'key' and 'value' fields.")
        key_id = None
        for field in self.content.customFieldsManager.field:
            if field.name == kv['key']:
                key_id = field.key
                break
        if not key_id:
            self.module.fail_json(msg='Unable to find custom value key %s' % kv['key'])
        if kv['key'] not in facts['customvalues'] or facts['customvalues'][kv['key']] != kv['value']:
            self.content.customFieldsManager.SetField(entity=vm_obj, key=key_id, value=kv['value'])
            self.change_detected = True