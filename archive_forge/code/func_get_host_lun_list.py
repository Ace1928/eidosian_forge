from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def get_host_lun_list(self, host_details):
    """ Get luns attached to host"""
    host_luns_list = []
    if host_details and host_details.host_luns is not None:
        for lun in host_details.host_luns.lun:
            host_lun = {'name': lun.name, 'id': lun.id}
            host_luns_list.append(host_lun)
    return host_luns_list