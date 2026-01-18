from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def portset_get_iter(self):
    """
        Compose NaElement object to query current portset using vserver, portset-name and portset-type parameters
        :return: NaElement object for portset-get-iter with query
        """
    portset_get = netapp_utils.zapi.NaElement('portset-get-iter')
    query = netapp_utils.zapi.NaElement('query')
    portset_info = netapp_utils.zapi.NaElement('portset-info')
    portset_info.add_new_child('vserver', self.parameters['vserver'])
    portset_info.add_new_child('portset-name', self.parameters['resource_name'])
    if self.parameters.get('portset_type'):
        portset_info.add_new_child('portset-type', self.parameters['portset_type'])
    query.add_child_elem(portset_info)
    portset_get.add_child_elem(query)
    return portset_get