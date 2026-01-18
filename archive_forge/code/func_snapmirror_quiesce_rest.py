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
def snapmirror_quiesce_rest(self):
    """
        SnapMirror quiesce using REST
        """
    if self.parameters['current_mirror_state'] == 'paused' or self.parameters['current_mirror_state'] == 'broken_off' or self.parameters['current_transfer_status'] == 'transferring':
        return
    self.snapmirror_mod_init_resync_break_quiesce_resume_rest(state='paused')
    self.wait_for_quiesced_status()