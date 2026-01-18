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
def customize_advanced_settings(self, vm_obj, config_spec):
    if not self.params['advanced_settings']:
        return
    vm_custom_spec = config_spec
    vm_custom_spec.extraConfig = []
    changed = False
    facts = self.gather_facts(vm_obj)
    for kv in self.params['advanced_settings']:
        if 'key' not in kv or 'value' not in kv:
            self.module.exit_json(msg="advanced_settings items required both 'key' and 'value' fields.")
        if isinstance(kv['value'], (bool, int)):
            specifiedvalue = str(kv['value']).upper()
            comparisonvalue = facts['advanced_settings'].get(kv['key'], '').upper()
        else:
            specifiedvalue = kv['value']
            comparisonvalue = facts['advanced_settings'].get(kv['key'], '')
        if kv['key'] not in facts['advanced_settings'] and kv['value'] != '' or comparisonvalue != specifiedvalue:
            option = vim.option.OptionValue()
            option.key = kv['key']
            option.value = specifiedvalue
            vm_custom_spec.extraConfig.append(option)
            changed = True
        if changed:
            self.change_detected = True