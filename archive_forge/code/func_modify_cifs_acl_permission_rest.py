from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_cifs_acl_permission_rest(self, current):
    """
        Change permission for the given CIFS share/user-group with rest API.
        """
    if not self.use_rest:
        return self.modify_cifs_acl_permission()
    body = {'permission': self.parameters.get('permission')}
    api = 'protocols/cifs/shares/%s/%s/acls/%s/%s' % (current['uuid'], self.parameters.get('share_name'), self.parameters.get('user_or_group'), current.get('type'))
    dummy, error = rest_generic.patch_async(self.rest_api, api, None, body)
    if error is not None:
        self.module.fail_json(msg='Error modifying cifs share ACL permission: %s' % error)