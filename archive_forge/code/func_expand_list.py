from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_list(items, fields, class_name):
    if items is not None:
        new_objects_list = []
        for item in items:
            new_obj = expand_fields(fields, item, class_name)
            new_objects_list.append(new_obj)
        return new_objects_list