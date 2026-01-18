from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_capacity(eg, module, is_update, do_not_update):
    eg_capacity = expand_fields(capacity_fields, module.params, 'Capacity')
    if is_update is True:
        delattr(eg_capacity, 'unit')
        if 'target' in do_not_update:
            delattr(eg_capacity, 'target')
    eg.capacity = eg_capacity