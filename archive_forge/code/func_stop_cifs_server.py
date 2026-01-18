from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def stop_cifs_server(self):
    """
        RModify the cifs_server.
        """
    cifs_server_modify = netapp_utils.zapi.NaElement.create_node_with_children('cifs-server-stop')
    try:
        self.server.invoke_successfully(cifs_server_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as e:
        self.module.fail_json(msg='Error modifying cifs_server %s: %s' % (self.parameters['cifs_server_name'], to_native(e)), exception=traceback.format_exc())