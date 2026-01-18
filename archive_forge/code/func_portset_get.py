from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def portset_get(self):
    """
        Get current portset info
        :return: List of current ports if query successful, else return []
        """
    portset_get_iter = self.portset_get_iter()
    result, ports = (None, [])
    try:
        result = self.server.invoke_successfully(portset_get_iter, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching portset %s: %s' % (self.parameters['resource_name'], to_native(error)), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0:
        portset_get_info = result.get_child_by_name('attributes-list').get_child_by_name('portset-info')
        if int(portset_get_info.get_child_content('portset-port-total')) > 0:
            port_info = portset_get_info.get_child_by_name('portset-port-info')
            ports = [port.get_content() for port in port_info.get_children()]
    return ports