from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def enable_volume_efficiency(self):
    """
        Enables Volume efficiency for a given volume by path
        """
    sis_enable = netapp_utils.zapi.NaElement('sis-enable')
    sis_enable.add_new_child('path', self.parameters['path'])
    try:
        self.server.invoke_successfully(sis_enable, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error enabling storage efficiency for path %s on vserver %s: %s' % (self.parameters['path'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())