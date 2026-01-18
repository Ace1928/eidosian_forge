from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def get_hosts_dict(snapshot):
    """ This method creates a dictionary, with host as key and
        allowed access as value """
    hosts_dict = {}
    LOG.info('Inside get_hosts_dict')
    if not snapshot.host_access:
        return hosts_dict
    for host_access_obj in snapshot.host_access:
        hosts_dict[host_access_obj.host] = host_access_obj.allowed_access
    return hosts_dict