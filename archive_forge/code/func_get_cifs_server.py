from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cifs_server(self):
    """
        Return details about the CIFS-server
        :param:
            name : Name of the name of the cifs_server

        :return: Details about the cifs_server. None if not found.
        :rtype: dict
        """
    cifs_server_info = netapp_utils.zapi.NaElement('cifs-server-get-iter')
    cifs_server_attributes = netapp_utils.zapi.NaElement('cifs-server-config')
    cifs_server_attributes.add_new_child('cifs-server', self.parameters['cifs_server_name'])
    cifs_server_attributes.add_new_child('vserver', self.parameters['vserver'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(cifs_server_attributes)
    cifs_server_info.add_child_elem(query)
    result = self.server.invoke_successfully(cifs_server_info, True)
    return_value = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        cifs_server_attributes = result.get_child_by_name('attributes-list').get_child_by_name('cifs-server-config')
        service_state = cifs_server_attributes.get_child_content('administrative-status')
        return_value = {'cifs_server_name': self.parameters['cifs_server_name'], 'service_state': 'started' if service_state == 'up' else 'stopped'}
    return return_value