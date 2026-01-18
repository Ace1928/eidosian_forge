from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_fcp(self):
    if self.use_rest:
        return self.get_fcp_rest()
    fcp_obj = netapp_utils.zapi.NaElement('fcp-service-get-iter')
    fcp_info = netapp_utils.zapi.NaElement('fcp-service-info')
    fcp_info.add_new_child('vserver', self.parameters['vserver'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(fcp_info)
    fcp_obj.add_child_elem(query)
    result = self.server.invoke_successfully(fcp_obj, True)
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        return True
    else:
        return False