from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_ipspace(self, name=None):
    """
        Fetch details if ipspace exists
        :param name: Name of the ipspace to be fetched
        :return:
            Dictionary of current details if ipspace found
            None if ipspace is not found
        """
    if name is None:
        name = self.parameters['name']
    if self.use_rest:
        api = 'network/ipspaces'
        query = {'name': name, 'fields': 'uuid'}
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error getting ipspace %s: %s' % (name, error))
        if record:
            self.uuid = record['uuid']
            return record
        return None
    else:
        ipspace_get = self.ipspace_get_iter(name)
        if ipspace_get and ipspace_get.get_child_by_name('num-records') and (int(ipspace_get.get_child_content('num-records')) >= 1):
            current_ipspace = dict()
            attr_list = ipspace_get.get_child_by_name('attributes-list')
            attr = attr_list.get_child_by_name('net-ipspaces-info')
            current_ipspace['name'] = attr.get_child_content('ipspace')
            return current_ipspace
        return None