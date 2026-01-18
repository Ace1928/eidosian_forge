from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_cg_snapshot_rest(self):
    """Create CG snapshot"""
    api = '/application/consistency-groups/%s/snapshots' % self.cg_uuid
    body = {'name': self.parameters['snapshot']}
    if self.parameters.get('snapmirror_label'):
        body['snapmirror_label'] = self.parameters['snapmirror_label']
    if self.parameters.get('comment'):
        body['comment'] = self.parameters['comment']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating consistency group snapshot %s: %s' % (self.parameters['snapshot'], to_native(error)), exception=traceback.format_exc())