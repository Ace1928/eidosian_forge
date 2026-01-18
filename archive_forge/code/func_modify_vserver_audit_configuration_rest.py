from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_vserver_audit_configuration_rest(self, modify):
    """
        Updates audit configuration.
        """
    body = {}
    if 'enabled' in modify:
        body['enabled'] = modify['enabled']
    else:
        body = self.create_vserver_audit_config_body_rest()
    api = 'protocols/audit'
    record, error = rest_generic.patch_async(self.rest_api, api, self.svm_uuid, body)
    if error:
        self.module.fail_json(msg='Error on modifying vserver audit configuration: %s' % error)