from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def modify_interface(self, modify, uuid=None, body=None):
    """
        Modify the interface.
        """
    if self.use_rest:
        return self.modify_interface_rest(uuid, body)
    migrate = {}
    modify_options = dict(modify)
    if modify_options.get('current_node') is not None:
        migrate['current_node'] = modify_options.pop('current_node')
    if modify_options.get('current_port') is not None:
        migrate['current_port'] = modify_options.pop('current_port')
    if modify_options:
        options = {'interface-name': self.parameters['interface_name'], 'vserver': self.parameters['vserver']}
        NetAppOntapInterface.set_options(options, modify_options)
        interface_modify = netapp_utils.zapi.NaElement.create_node_with_children('net-interface-modify', **options)
        try:
            self.server.invoke_successfully(interface_modify, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as err:
            self.module.fail_json(msg='Error modifying interface %s: %s' % (self.parameters['interface_name'], to_native(err)), exception=traceback.format_exc())
    if migrate:
        self.migrate_interface()