from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def migrate_interface(self):
    interface_migrate = netapp_utils.zapi.NaElement('net-interface-migrate')
    if self.parameters.get('current_node') is None:
        self.module.fail_json(msg='current_node must be set to migrate')
    interface_migrate.add_new_child('destination-node', self.parameters['current_node'])
    if self.parameters.get('current_port') is not None:
        interface_migrate.add_new_child('destination-port', self.parameters['current_port'])
    interface_migrate.add_new_child('lif', self.parameters['interface_name'])
    interface_migrate.add_new_child('vserver', self.parameters['vserver'])
    try:
        self.server.invoke_successfully(interface_migrate, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error migrating %s: %s' % (self.parameters['current_node'], to_native(error)), exception=traceback.format_exc())
    try:
        self.server.invoke_successfully(interface_migrate, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error migrating %s: %s' % (self.parameters['current_node'], to_native(error)), exception=traceback.format_exc())