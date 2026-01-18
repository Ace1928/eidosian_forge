from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
def check_node_has_active_drives(self, node_id=None):
    """
            Check if node has active drives attached to cluster
            :description: Validate if node have active drives in cluster

            :return: True or False
            :rtype: bool
        """
    if node_id is not None:
        cluster_drives = self.sfe.list_drives()
        for drive in cluster_drives.drives:
            if drive.node_id == node_id and drive.status == 'active':
                return True
    return False