from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def online_or_offline_adapter(self, status, adapter_name):
    """
        Bring a Fibre Channel target adapter offline/online.
        """
    if self.use_rest:
        return self.online_or_offline_adapter_rest(status, adapter_name)
    if status == 'down':
        adapter = netapp_utils.zapi.NaElement('fcp-adapter-config-down')
    elif status == 'up':
        adapter = netapp_utils.zapi.NaElement('fcp-adapter-config-up')
    adapter.add_new_child('fcp-adapter', adapter_name)
    adapter.add_new_child('node', self.parameters['node_name'])
    try:
        self.server.invoke_successfully(adapter, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as e:
        self.module.fail_json(msg='Error trying to %s fc-adapter %s: %s' % (status, adapter_name, to_native(e)), exception=traceback.format_exc())