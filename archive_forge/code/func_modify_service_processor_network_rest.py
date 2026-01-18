from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import time
def modify_service_processor_network_rest(self, modify):
    api = 'cluster/nodes'
    body = {'service_processor': {}}
    ipv4_or_ipv6_body = {}
    if self.parameters.get('gateway_ip_address'):
        ipv4_or_ipv6_body['gateway'] = self.parameters['gateway_ip_address']
    if self.parameters.get('netmask'):
        ipv4_or_ipv6_body['netmask'] = self.parameters['netmask']
    if self.parameters.get('prefix_length'):
        ipv4_or_ipv6_body['netmask'] = self.parameters['prefix_length']
    if self.parameters.get('ip_address'):
        ipv4_or_ipv6_body['address'] = self.parameters['ip_address']
    if ipv4_or_ipv6_body:
        body['service_processor'][self.ipv4_or_ipv6] = ipv4_or_ipv6_body
    if 'dhcp' in self.parameters:
        body['service_processor']['dhcp_enabled'] = True if self.parameters['dhcp'] == 'v4' else False
    elif ipv4_or_ipv6_body.get('gateway') and ipv4_or_ipv6_body.get('address') and ipv4_or_ipv6_body.get('netmask'):
        body['service_processor']['dhcp_enabled'] = False
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body)
    if error:
        self.module.fail_json(msg='Error modifying service processor network: %s' % error)
    if self.parameters.get('wait_for_completion'):
        retries = 25
        while self.is_sp_modified_rest(modify) is False and retries > 0:
            time.sleep(15)
            retries -= 1