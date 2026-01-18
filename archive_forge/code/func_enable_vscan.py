from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def enable_vscan(self, uuid=None):
    if self.use_rest:
        params = {'svm.name': self.parameters['vserver']}
        data = {'enabled': self.parameters['enable']}
        api = 'protocols/vscan/' + uuid
        dummy, error = self.rest_api.patch(api, data, params)
        if error is not None:
            self.module.fail_json(msg=error)
    else:
        vscan_status_obj = netapp_utils.zapi.NaElement('vscan-status-modify')
        vscan_status_obj.add_new_child('is-vscan-enabled', str(self.parameters['enable']))
        try:
            self.server.invoke_successfully(vscan_status_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error Enable/Disabling Vscan: %s' % to_native(error), exception=traceback.format_exc())