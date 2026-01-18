from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def create_export_policy(self):
    """
        Creates an export policy
        """
    if self.use_rest:
        return self.create_export_policy_rest()
    export_policy_create = netapp_utils.zapi.NaElement.create_node_with_children('export-policy-create', **{'policy-name': self.parameters['name']})
    try:
        self.server.invoke_successfully(export_policy_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating export policy %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())