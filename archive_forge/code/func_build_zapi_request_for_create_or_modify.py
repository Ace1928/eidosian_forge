from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def build_zapi_request_for_create_or_modify(self, zapi):
    simple_keys = ['gateway', 'ipspace', 'subnet']
    options = {'subnet-name': self.parameters.get('name')}
    if zapi == 'net-subnet-create':
        options['broadcast-domain'] = self.parameters.get('broadcast_domain')
        options['subnet'] = self.parameters.get('subnet')
        simple_keys.remove('subnet')
    for key in simple_keys:
        value = self.parameters.get(key)
        if value is not None:
            options[key] = value
    result = netapp_utils.zapi.NaElement.create_node_with_children(zapi, **options)
    if self.parameters.get('ip_ranges'):
        subnet_ips = netapp_utils.zapi.NaElement('ip-ranges')
        for ip_range in self.parameters.get('ip_ranges'):
            subnet_ips.add_new_child('ip-range', ip_range)
        result.add_child_elem(subnet_ips)
    return result