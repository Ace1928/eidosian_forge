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
def set_vapp_properties(self, property_spec):
    property_info = vim.vApp.PropertyInfo()
    property_info.classId = property_spec.get('classId')
    property_info.instanceId = property_spec.get('instanceId')
    property_info.id = property_spec.get('id')
    property_info.category = property_spec.get('category')
    property_info.label = property_spec.get('label')
    property_info.type = property_spec.get('type', 'string')
    property_info.userConfigurable = property_spec.get('userConfigurable', True)
    property_info.defaultValue = property_spec.get('defaultValue')
    property_info.value = property_spec.get('value', '')
    property_info.description = property_spec.get('description')
    return property_info