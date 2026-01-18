from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_port(self, port, zapi, action):
    """
        Add or remove an port to/from a portset
        """
    options = {'portset-name': self.parameters['name'], 'portset-port-name': port}
    portset_modify = netapp_utils.zapi.NaElement.create_node_with_children(zapi, **options)
    try:
        self.server.invoke_successfully(portset_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error %s port in portset %s: %s' % (action, self.parameters['name'], to_native(error)), exception=traceback.format_exc())