from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def check_custom_properties():
    if self.param('custom_properties'):
        current = []
        if entity.custom_properties:
            current = [(cp.name, cp.regexp, str(cp.value)) for cp in entity.custom_properties]
        passed = [(cp.get('name'), cp.get('regexp'), str(cp.get('value'))) for cp in self.param('custom_properties') if cp]
        return sorted(current) == sorted(passed)
    return True