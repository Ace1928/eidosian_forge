from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_unix_user_rest(self):
    """
        Creates the local UNIX user configuration for an SVM with rest API.
        """
    if not self.use_rest:
        return self.create_unix_user()
    body = {'svm.name': self.parameters.get('vserver')}
    for key in ('name', 'full_name', 'id', 'primary_gid'):
        if key in self.parameters:
            body[key] = self.parameters.get(key)
    api = 'name-services/unix-users'
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error is not None:
        self.module.fail_json(msg='Error on creating unix-user: %s' % error)