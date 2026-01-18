from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
def get_drives_listby_status(self, node_num_ids):
    """
            Capture list of drives based on status for a given node_id
            :description: Capture list of active, failed and available drives from a given node_id

            :return: None
        """
    self.active_drives = dict()
    self.available_drives = dict()
    self.other_drives = dict()
    self.all_drives = self.sfe.list_drives()
    for drive in self.all_drives.drives:
        if node_num_ids is None or drive.node_id in node_num_ids:
            if drive.status in ['active', 'failed']:
                self.active_drives[drive.serial] = drive.drive_id
            elif drive.status == 'available':
                self.available_drives[drive.serial] = drive.drive_id
            else:
                self.other_drives[drive.serial] = (drive.drive_id, drive.status)
    self.debug.append('available: %s' % self.available_drives)
    self.debug.append('active: %s' % self.active_drives)
    self.debug.append('other: %s' % self.other_drives)