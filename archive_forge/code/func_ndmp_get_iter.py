from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def ndmp_get_iter(self, uuid=None):
    """
        get current vserver ndmp attributes.
        :return: a dict of ndmp attributes.
        """
    if self.use_rest:
        data = dict()
        params = {'fields': 'authentication_types,enabled'}
        api = '/protocols/ndmp/svms/' + uuid
        message, error = self.rest_api.get(api, params)
        data['enable'] = message['enabled']
        data['authtype'] = message['authentication_types']
        if error:
            self.module.fail_json(msg=error)
        return data
    else:
        ndmp_get = netapp_utils.zapi.NaElement('ndmp-vserver-attributes-get-iter')
        query = netapp_utils.zapi.NaElement('query')
        ndmp_info = netapp_utils.zapi.NaElement('ndmp-vserver-attributes-info')
        ndmp_info.add_new_child('vserver', self.parameters['vserver'])
        query.add_child_elem(ndmp_info)
        ndmp_get.add_child_elem(query)
        ndmp_details = dict()
        try:
            result = self.server.invoke_successfully(ndmp_get, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error fetching ndmp from %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
        if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0:
            ndmp_attributes = result.get_child_by_name('attributes-list').get_child_by_name('ndmp-vserver-attributes-info')
            self.get_ndmp_details(ndmp_details, ndmp_attributes)
        return ndmp_details