from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_unix_user_rest(self, modify, current=None):
    """
        Updates UNIX user information for the specified user and SVM with rest API.
        """
    if not self.use_rest:
        return self.modify_unix_user(modify)
    query = {'svm.name': self.parameters.get('vserver')}
    body = {}
    for key in ('full_name', 'id', 'primary_gid'):
        if key in modify:
            body[key] = modify[key]
    api = 'name-services/unix-users/%s' % current['svm']['uuid']
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.parameters['name'], body, query)
    if error is not None:
        self.module.fail_json(msg='Error on modifying unix-user: %s' % error)