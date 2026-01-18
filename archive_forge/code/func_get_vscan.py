from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def get_vscan(self):
    if self.use_rest:
        params = {'fields': 'svm,enabled', 'svm.name': self.parameters['vserver']}
        api = 'protocols/vscan'
        message, error = self.rest_api.get(api, params)
        if error:
            self.module.fail_json(msg=error)
        return message['records'][0]
    else:
        vscan_status_iter = netapp_utils.zapi.NaElement('vscan-status-get-iter')
        vscan_status_info = netapp_utils.zapi.NaElement('vscan-status-info')
        vscan_status_info.add_new_child('vserver', self.parameters['vserver'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(vscan_status_info)
        vscan_status_iter.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(vscan_status_iter, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error getting Vscan info for Vserver %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
        if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
            return result.get_child_by_name('attributes-list').get_child_by_name('vscan-status-info')