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
def obj_has_parent(self, obj, parent):
    if obj is None and parent is None:
        raise AssertionError()
    current_parent = obj
    while True:
        if current_parent.name == parent.name:
            return True
        moid = current_parent._moId
        if moid in ['group-d1', 'ha-folder-root']:
            return False
        current_parent = current_parent.parent
        if current_parent is None:
            return False