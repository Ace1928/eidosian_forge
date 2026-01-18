from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def modify_public_key(self, current, modify):
    api = 'security/authentication/publickeys/%s/%s/%d' % (current['owner']['uuid'], current['account'], current['index'])
    body = {}
    modify_copy = dict(modify)
    for key in modify:
        if key in ('comment', 'public_key'):
            body[key] = modify_copy.pop(key)
    if modify_copy:
        msg = 'Error: attributes not supported in modify: %s' % modify_copy
        self.module.fail_json(msg=msg)
    if not body:
        msg = 'Error: nothing to change - modify called with: %s' % modify
        self.module.fail_json(msg=msg)
    if 'public_key' not in body:
        body['public_key'] = current['public_key']
    dummy, error = self.rest_api.patch(api, body)
    if error:
        msg = 'Error in modify_public_key: %s' % error
        self.module.fail_json(msg=msg)