from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_vserver_audit_configuration_rest(self):
    """
        Creates an audit configuration.
        """
    api = 'protocols/audit'
    body = self.create_vserver_audit_config_body_rest()
    if 'vserver' in self.parameters:
        body['svm.name'] = self.parameters.get('vserver')
    if 'enabled' in self.parameters:
        body['enabled'] = self.parameters['enabled']
    record, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error on creating vserver audit configuration: %s' % error)