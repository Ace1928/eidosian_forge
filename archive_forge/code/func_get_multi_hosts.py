from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def get_multi_hosts(module):
    """Return True is all hosts exist"""
    hosts = []
    array = get_array(module)
    for host_num in range(module.params['start'], module.params['count'] + module.params['start']):
        if module.params['suffix']:
            hosts.append(module.params['name'] + str(host_num).zfill(module.params['digits']) + module.params['suffix'])
        else:
            hosts.append(module.params['name'] + str(host_num).zfill(module.params['digits']))
    return bool(array.get_hosts(names=hosts).status_code == 200)