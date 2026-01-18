from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_subsystem(self):
    """
        Get current subsystem details
        :return: dict if subsystem exists, None otherwise
        """
    if self.use_rest:
        return self.get_subsystem_rest()
    result = self.get_zapi_info('nvme-subsystem-get-iter', 'nvme-subsystem-info')
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        return True
    return None