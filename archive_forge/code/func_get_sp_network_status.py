from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import time
def get_sp_network_status(self):
    """
        Return status of service processor network
        :param:
            name : name of the node
        :return: Status of the service processor network
        :rtype: dict
        """
    spn_get_iter = netapp_utils.zapi.NaElement('service-processor-network-get-iter')
    query_info = {'query': {'service-processor-network-info': {'node': self.parameters['node'], 'address-type': self.parameters['address_type']}}}
    spn_get_iter.translate_struct(query_info)
    try:
        result = self.server.invoke_successfully(spn_get_iter, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching service processor network status for %s: %s' % (self.parameters['node'], to_native(error)), exception=traceback.format_exc())
    if int(result['num-records']) >= 1:
        sp_attr_info = result['attributes-list']['service-processor-network-info']
        return sp_attr_info.get_child_content('setup-status')
    return None