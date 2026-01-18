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
def configure_vapp_properties(self, vm_obj):
    if not self.params['vapp_properties']:
        return
    for x in self.params['vapp_properties']:
        if not x.get('id'):
            self.module.fail_json(msg='id is required to set vApp property')
    new_vmconfig_spec = vim.vApp.VmConfigSpec()
    if vm_obj:
        orig_spec = vm_obj.config.vAppConfig
        vapp_properties_current = dict(((x.id, x) for x in orig_spec.property))
        vapp_properties_to_change = dict(((x['id'], x) for x in self.params['vapp_properties']))
        all_keys = [x.key for x in orig_spec.property]
        new_property_index = max(all_keys) + 1 if all_keys else 0
        for property_id, property_spec in vapp_properties_to_change.items():
            is_property_changed = False
            new_vapp_property_spec = vim.vApp.PropertySpec()
            if property_id in vapp_properties_current:
                if property_spec.get('operation') == 'remove':
                    new_vapp_property_spec.operation = 'remove'
                    new_vapp_property_spec.removeKey = vapp_properties_current[property_id].key
                    is_property_changed = True
                else:
                    new_vapp_property_spec.operation = 'edit'
                    new_vapp_property_spec.info = vapp_properties_current[property_id]
                    try:
                        for property_name, property_value in property_spec.items():
                            if property_name == 'operation':
                                continue
                            if getattr(new_vapp_property_spec.info, property_name) != property_value:
                                setattr(new_vapp_property_spec.info, property_name, property_value)
                                is_property_changed = True
                    except Exception as e:
                        msg = "Failed to set vApp property field='%s' and value='%s'. Error: %s" % (property_name, property_value, to_text(e))
                        self.module.fail_json(msg=msg)
            else:
                if property_spec.get('operation') == 'remove':
                    continue
                new_vapp_property_spec.operation = 'add'
                property_info = self.set_vapp_properties(property_spec)
                new_vapp_property_spec.info = property_info
                new_vapp_property_spec.info.key = new_property_index
                new_property_index += 1
                is_property_changed = True
            if is_property_changed:
                new_vmconfig_spec.property.append(new_vapp_property_spec)
    else:
        all_keys = [x.key for x in new_vmconfig_spec.property]
        new_property_index = max(all_keys) + 1 if all_keys else 0
        vapp_properties_to_change = dict(((x['id'], x) for x in self.params['vapp_properties']))
        is_property_changed = False
        for property_id, property_spec in vapp_properties_to_change.items():
            new_vapp_property_spec = vim.vApp.PropertySpec()
            new_vapp_property_spec.operation = 'add'
            property_info = self.set_vapp_properties(property_spec)
            new_vapp_property_spec.info = property_info
            new_vapp_property_spec.info.key = new_property_index
            new_property_index += 1
            is_property_changed = True
            if is_property_changed:
                new_vmconfig_spec.property.append(new_vapp_property_spec)
    if new_vmconfig_spec.property:
        self.configspec.vAppConfig = new_vmconfig_spec
        self.change_detected = True