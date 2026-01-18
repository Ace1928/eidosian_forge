from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def get_volume_style(self, current):
    """Get volume style, infinite or standard flexvol"""
    if current is not None:
        return current.get('style_extended')
    if self.parameters.get('aggr_list') or self.parameters.get('aggr_list_multiplier') or self.parameters.get('auto_provision_as'):
        if self.use_rest and self.parameters.get('auto_provision_as') and (self.parameters.get('aggr_list_multiplier') is None):
            self.parameters['aggr_list_multiplier'] = 1
        return 'flexgroup'
    return None