from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def send_zapi_message(self, params, node_name):
    params['node-name'] = node_name
    send_message = netapp_utils.zapi.NaElement.create_node_with_children('autosupport-invoke', **params)
    try:
        self.server.invoke_successfully(send_message, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error on sending autosupport message to node %s: %s.' % (node_name, to_native(error)), exception=traceback.format_exc())