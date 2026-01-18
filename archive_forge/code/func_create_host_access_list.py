from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def create_host_access_list(snapshot, host, host_state):
    """ This method creates a List of dictionaries which will be used
        to modify the list of hosts mapped to a snapshot """
    host_access_list = []
    hosts_dict = get_hosts_dict(snapshot)
    if not hosts_dict:
        return None
    if to_update_host_list(snapshot, host, host_state):
        if host_state == 'mapped':
            return None
        for snap_host in list(hosts_dict.keys()):
            if snap_host != host:
                access_dict = {'host': snap_host, 'allowedAccess': hosts_dict[snap_host]}
                host_access_list.append(access_dict)
    return host_access_list