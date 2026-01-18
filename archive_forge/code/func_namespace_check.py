from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def namespace_check(self):
    command = ['list', '-R']
    out = self.pmem_run_ndctl(command)
    if not out:
        return 'Available region(s) is not in this system.'
    region = json.loads(out)
    aligns = self.pmem_get_region_align_size(region)
    if len(aligns) != 1:
        return 'Not supported the regions whose alignment size is different.'
    available_size = self.pmem_get_available_region_size(region)
    types = self.pmem_get_available_region_type(region)
    for ns in self.namespace:
        if ns['size']:
            try:
                size_byte = human_to_bytes(ns['size'])
            except ValueError:
                return 'The format of size: NNN TB|GB|MB|KB|T|G|M|K|B'
            if size_byte % aligns[0] != 0:
                return 'size: %s should be align with %d' % (ns['size'], aligns[0])
            is_space_enough = False
            for i, avail in enumerate(available_size):
                if avail > size_byte:
                    available_size[i] -= size_byte
                    is_space_enough = True
                    break
            if is_space_enough is False:
                return 'There is not available region for size: %s' % ns['size']
            ns['size_byte'] = size_byte
        elif len(self.namespace) != 1:
            return 'size option is required to configure multiple namespaces'
        if ns['type'] not in types:
            return 'type %s is not supported in this system. Supported type: %s' % (ns['type'], types)
    return None