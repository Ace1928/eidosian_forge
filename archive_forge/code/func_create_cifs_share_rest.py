from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_cifs_share_rest(self):
    """
        create CIFS share with rest API.
        """
    if not self.use_rest:
        return self.create_cifs_share()
    body = self.create_modify_body_rest()
    if 'vserver' in self.parameters:
        body['svm.name'] = self.parameters['vserver']
    if 'name' in self.parameters:
        body['name'] = self.parameters['name']
    if 'path' in self.parameters:
        body['path'] = self.parameters['path']
    api = 'protocols/cifs/shares'
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error is not None:
        self.module.fail_json(msg='Error on creating cifs shares: %s' % error)