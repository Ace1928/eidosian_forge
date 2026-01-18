from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def patch_cluster_dns(self):
    api = 'cluster'
    body = {'dns_domains': self.parameters['domains'], 'name_servers': self.parameters['nameservers']}
    if self.parameters.get('skip_validation'):
        self.module.warn('skip_validation is ignored for cluster DNS operations in REST.')
    dummy, error = rest_generic.patch_async(self.rest_api, api, None, body)
    if error:
        self.module.fail_json(msg='Error updating cluster DNS options: %s' % error)