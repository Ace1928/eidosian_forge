from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_snapmirror(self, relationship_type, mirror_state):
    """
        Delete a SnapMirror relationship
        #1. Quiesce the SnapMirror relationship at destination
        #2. Break the SnapMirror relationship at the destination
        #3. Release the SnapMirror at source
        #4. Delete SnapMirror at destination
        """
    if relationship_type not in ['load_sharing', 'vault'] and mirror_state not in ['uninitialized', 'broken-off', 'broken_off']:
        self.snapmirror_break(before_delete=True)
    if self.parameters.get('peer_options') is not None and self.parameters.get('connection_type') != 'elementsw_ontap' and (not self.use_rest):
        self.set_source_cluster_connection()
        if self.get_destination():
            self.snapmirror_release()
    self.snapmirror_delete()