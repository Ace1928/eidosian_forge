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
def get_svm_peer(self, source_svm, destination_svm):
    if self.use_rest:
        api = 'svm/peers'
        query = {'name': source_svm, 'svm.name': destination_svm}
        record, error = rest_generic.get_one_record(self.rest_api, api, query, fields='peer')
        if error:
            self.module.fail_json(msg='Error retrieving SVM peer: %s' % error)
        if record:
            return (self.na_helper.safe_get(record, ['peer', 'svm', 'name']), self.na_helper.safe_get(record, ['peer', 'cluster', 'name']))
    else:
        query = {'query': {'vserver-peer-info': {'peer-vserver': source_svm, 'vserver': destination_svm}}}
        get_request = netapp_utils.zapi.NaElement('vserver-peer-get-iter')
        get_request.translate_struct(query)
        try:
            result = self.server.invoke_successfully(get_request, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error fetching vserver peer info: %s' % to_native(error), exception=traceback.format_exc())
        if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0:
            info = result.get_child_by_name('attributes-list').get_child_by_name('vserver-peer-info')
            return (info['remote-vserver-name'], info['peer-cluster'])
    return (None, None)