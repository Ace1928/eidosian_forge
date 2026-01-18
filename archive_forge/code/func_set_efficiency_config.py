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
def set_efficiency_config(self):
    """Set efficiency policy and compression attributes"""
    options = {'path': '/vol/' + self.parameters['name']}
    efficiency_enable = netapp_utils.zapi.NaElement.create_node_with_children('sis-enable', **options)
    try:
        self.server.invoke_successfully(efficiency_enable, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        if to_native(error.code) != '40043':
            self.wrap_fail_json(msg='Error enable efficiency on volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    self.set_efficiency_attributes(options)
    efficiency_start = netapp_utils.zapi.NaElement.create_node_with_children('sis-set-config', **options)
    try:
        self.server.invoke_successfully(efficiency_start, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.wrap_fail_json(msg='Error setting up efficiency attributes on volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())