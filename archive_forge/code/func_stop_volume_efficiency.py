from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def stop_volume_efficiency(self):
    """
        Stops volume efficiency for a given flex volume by path
        """
    sis_stop = netapp_utils.zapi.NaElement('sis-stop')
    sis_stop.add_new_child('path', self.parameters['path'])
    if 'stop_ve_all_operations' in self.parameters:
        sis_stop.add_new_child('all-operations', self.na_helper.get_value_for_bool(False, self.parameters['stop_ve_all_operations']))
    try:
        self.server.invoke_successfully(sis_stop, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error stopping storage efficiency for path %s on vserver %s: %s' % (self.parameters['path'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())