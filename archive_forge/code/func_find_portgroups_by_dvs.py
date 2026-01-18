from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils.six.moves.urllib.parse import unquote
def find_portgroups_by_dvs(self, pgl, dvs):
    obj = []
    for c in pgl:
        if dvs in c.config.distributedVirtualSwitch.name:
            obj.append(c)
    return obj