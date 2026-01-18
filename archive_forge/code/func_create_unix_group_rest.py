from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_unix_group_rest(self):
    """
        Creates the local UNIX group configuration for the specified SVM.
        Group name and group ID are mandatory parameters.
        """
    if not self.use_rest:
        return self.create_unix_group()
    body = {'svm.name': self.parameters.get('vserver')}
    if 'name' in self.parameters:
        body['name'] = self.parameters['name']
    if 'id' in self.parameters:
        body['id'] = self.parameters['id']
    if 'skip_name_validation' in self.parameters:
        body['skip_name_validation'] = self.parameters['skip_name_validation']
    api = 'name-services/unix-groups'
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error is not None:
        self.module.fail_json(msg='Error creating UNIX group: %s' % error)
    if self.parameters.get('users') is not None:
        self.modify_users_in_group_rest()