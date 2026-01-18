from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def create_snapshot_details_dict(snapshot):
    """ Add name and id of storage resource and hosts to snapshot details """
    snapshot_dict = snapshot._get_properties()
    del snapshot_dict['storage_resource']
    del snapshot_dict['host_access']
    snapshot_dict['hosts_list'] = get_hosts_list(get_hosts_dict(snapshot))
    snapshot_dict['storage_resource_name'] = snapshot.storage_resource.name
    snapshot_dict['storage_resource_id'] = snapshot.storage_resource.id
    return snapshot_dict