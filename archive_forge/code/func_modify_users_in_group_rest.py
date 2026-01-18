from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_users_in_group_rest(self, current=None):
    """
        Add/delete one or many users in a UNIX group
        """
    body = {'records': []}
    if not current:
        current = self.get_unix_group_rest()
    current_users = current['users'] or []
    expect_users = self.parameters.get('users')
    users_to_remove = list(set(current_users) - set(expect_users))
    users_to_add = list(set(expect_users) - set(current_users))
    if len(users_to_add) > 0:
        body['records'] = [{'name': user} for user in users_to_add]
        if 'skip_name_validation' in self.parameters:
            body['skip_name_validation'] = self.parameters['skip_name_validation']
        api = 'name-services/unix-groups/%s/%s/users' % (current['svm']['uuid'], current['name'])
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error is not None:
            self.module.fail_json(msg='Error Adding user to UNIX group: %s' % error)
    if len(users_to_remove) > 0:
        for user in users_to_remove:
            api = 'name-services/unix-groups/%s/%s/users' % (current['svm']['uuid'], current['name'])
            dummy, error = rest_generic.delete_async(self.rest_api, api, user, body=None)
            if error is not None:
                self.module.fail_json(msg='Error removing user from UNIX group: %s' % error)