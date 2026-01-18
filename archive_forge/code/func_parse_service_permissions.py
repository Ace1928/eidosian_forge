from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
def parse_service_permissions(service):
    perm_dict = service['PERMISSIONS']
    '\n    This is the structure of the \'PERMISSIONS\' dictionary:\n\n   "PERMISSIONS": {\n                      "OWNER_U": "1",\n                      "OWNER_M": "1",\n                      "OWNER_A": "0",\n                      "GROUP_U": "0",\n                      "GROUP_M": "0",\n                      "GROUP_A": "0",\n                      "OTHER_U": "0",\n                      "OTHER_M": "0",\n                      "OTHER_A": "0"\n                    }\n    '
    owner_octal = int(perm_dict['OWNER_U']) * 4 + int(perm_dict['OWNER_M']) * 2 + int(perm_dict['OWNER_A'])
    group_octal = int(perm_dict['GROUP_U']) * 4 + int(perm_dict['GROUP_M']) * 2 + int(perm_dict['GROUP_A'])
    other_octal = int(perm_dict['OTHER_U']) * 4 + int(perm_dict['OTHER_M']) * 2 + int(perm_dict['OTHER_A'])
    permissions = str(owner_octal) + str(group_octal) + str(other_octal)
    return permissions