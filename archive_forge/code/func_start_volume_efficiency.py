from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def start_volume_efficiency(self):
    """
        Starts volume efficiency for a given flex volume by path
        """
    sis_start = netapp_utils.zapi.NaElement('sis-start')
    sis_start.add_new_child('path', self.parameters['path'])
    if 'start_ve_scan_all' in self.parameters:
        sis_start.add_new_child('scan-all', self.na_helper.get_value_for_bool(False, self.parameters['start_ve_scan_all']))
    if 'start_ve_build_metadata' in self.parameters:
        sis_start.add_new_child('build-metadata', self.na_helper.get_value_for_bool(False, self.parameters['start_ve_build_metadata']))
    if 'start_ve_delete_checkpoint' in self.parameters:
        sis_start.add_new_child('delete-checkpoint', self.na_helper.get_value_for_bool(False, self.parameters['start_ve_delete_checkpoint']))
    if 'start_ve_queue_operation' in self.parameters:
        sis_start.add_new_child('queue-operation', self.na_helper.get_value_for_bool(False, self.parameters['start_ve_queue_operation']))
    if 'start_ve_scan_old_data' in self.parameters:
        sis_start.add_new_child('scan', self.na_helper.get_value_for_bool(False, self.parameters['start_ve_scan_old_data']))
    if 'start_ve_qos_policy' in self.parameters:
        sis_start.add_new_child('qos-policy', self.parameters['start_ve_qos_policy'])
    try:
        self.server.invoke_successfully(sis_start, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error starting storage efficiency for path %s on vserver %s: %s' % (self.parameters['path'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())