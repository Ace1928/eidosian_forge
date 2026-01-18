from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def create_fs_snapshot_details_dict(self, fs_snapshot):
    """ Add name and id of storage resource to filesystem snapshot
            details """
    snapshot_dict = fs_snapshot._get_properties()
    del snapshot_dict['storage_resource']
    snapshot_dict['filesystem_name'] = fs_snapshot.storage_resource.name
    snapshot_dict['filesystem_id'] = fs_snapshot.storage_resource.filesystem.id
    obj_fs = self.unity_conn.get_filesystem(id=fs_snapshot.storage_resource.filesystem.id)
    if obj_fs and obj_fs.existed:
        snapshot_dict['nas_server_name'] = obj_fs.nas_server[0].name
        snapshot_dict['nas_server_id'] = obj_fs.nas_server[0].id
    return snapshot_dict