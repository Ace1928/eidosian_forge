from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def modify_dns_rest(self, dns_attrs):
    if self.is_cluster:
        return self.patch_cluster_dns()
    body = {}
    if dns_attrs['nameservers'] != self.parameters['nameservers']:
        body['servers'] = self.parameters['nameservers']
    if dns_attrs['domains'] != self.parameters['domains']:
        body['domains'] = self.parameters['domains']
    if 'skip_validation' in self.parameters:
        body['skip_config_validation'] = self.parameters['skip_validation']
    api = 'name-services/dns'
    dummy, error = rest_generic.patch_async(self.rest_api, api, dns_attrs['uuid'], body)
    if error:
        self.module.fail_json(msg='Error modifying DNS configuration: %s' % error)