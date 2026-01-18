from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def add_subsystem_host_map(self, data, type):
    """
        Add a NVME Subsystem host/map
        :param: data: list of hosts/paths to be added
        :param: type: hosts/paths
        """
    if type == 'hosts':
        zapi_add, zapi_type = ('nvme-subsystem-host-add', 'host-nqn')
    elif type == 'paths':
        zapi_add, zapi_type = ('nvme-subsystem-map-add', 'path')
    for item in data:
        options = {'subsystem': self.parameters['subsystem'], zapi_type: item}
        subsystem_add = netapp_utils.zapi.NaElement.create_node_with_children(zapi_add, **options)
        try:
            self.server.invoke_successfully(subsystem_add, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error adding %s for subsystem %s: %s' % (item, self.parameters.get('subsystem'), to_native(error)), exception=traceback.format_exc())