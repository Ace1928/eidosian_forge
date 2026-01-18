from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
def get_drive_id(self, drive_id, node_num_ids):
    """
            Get Drive ID
            :description: Find and retrieve drive_id from the active cluster
            Assumes self.all_drives is already populated

            :return: node_id (None if not found)
            :rtype: node_id
        """
    for drive in self.all_drives.drives:
        if drive_id == str(drive.drive_id):
            break
        if drive_id == drive.serial:
            break
    else:
        self.module.fail_json(msg='unable to find drive for drive_id=%s.  Debug=%s' % (drive_id, self.debug))
    if node_num_ids and drive.node_id not in node_num_ids:
        self.module.fail_json(msg='drive for drive_id=%s belongs to another node, with node_id=%d.  Debug=%s' % (drive_id, drive.node_id, self.debug))
    return (drive.drive_id, drive.status)