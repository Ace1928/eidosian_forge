from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_cifs_share_rest(self):
    """
        delete CIFS share with rest API.
        """
    if not self.use_rest:
        return self.delete_cifs_share()
    body = {'name': self.parameters.get('name')}
    api = 'protocols/cifs/shares'
    dummy, error = rest_generic.delete_async(self.rest_api, api, self.svm_uuid, body)
    if error is not None:
        self.module.fail_json(msg=' Error on deleting cifs shares: %s' % error)