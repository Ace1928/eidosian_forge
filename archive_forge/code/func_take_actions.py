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
def take_actions(self, actions, current, modify):
    if 'restore' in actions:
        self.snapmirror_restore()
    if 'create' in actions:
        self.snapmirror_create()
    if 'abort' in actions:
        self.snapmirror_abort()
        self.wait_for_idle_status()
    if 'delete' in actions:
        self.delete_snapmirror(current['relationship_type'], current['mirror_state'])
    if 'modify' in actions:
        self.snapmirror_modify(modify)
    if 'break' in actions:
        self.snapmirror_break()
    if 'initialize' in actions:
        self.snapmirror_initialize(current)
    if 'resume' in actions:
        self.snapmirror_resume()
    if 'resync' in actions:
        self.snapmirror_resync()