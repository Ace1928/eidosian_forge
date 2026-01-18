from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def form_create_modify_body_rest(self, params=None):
    if params is None:
        params = self.parameters
    body = {'name': self.parameters['name']}
    if params.get('broadcast_domain'):
        body['broadcast_domain.name'] = params['broadcast_domain']
    if params.get('subnet'):
        if '/' not in params['subnet']:
            self.module.fail_json(msg='Error: Invalid value specified for subnet %s' % params['subnet'])
        body['subnet.address'] = params['subnet'].split('/')[0]
        body['subnet.netmask'] = params['subnet'].split('/')[1]
    if params.get('gateway'):
        body['gateway'] = params['gateway']
    if params.get('ipspace'):
        body['ipspace.name'] = params['ipspace']
    ip_ranges = []
    for each_range in params.get('ip_ranges', []):
        if '-' in each_range:
            ip_ranges.append({'start': each_range.split('-')[0], 'end': each_range.split('-')[1]})
        else:
            ip_ranges.append({'start': each_range, 'end': each_range})
    if ip_ranges or params.get('ip_ranges') == []:
        body['ip_ranges'] = ip_ranges
    return body