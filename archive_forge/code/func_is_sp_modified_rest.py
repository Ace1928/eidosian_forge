from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import time
def is_sp_modified_rest(self, modify):
    current = self.get_service_processor_network_rest()
    if current is None:
        return False
    for sp_option in modify:
        if modify[sp_option] != current[sp_option]:
            return False
    return True