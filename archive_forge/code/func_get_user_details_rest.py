from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def get_user_details_rest(self, name, owner_uuid):
    query = {'fields': 'role,applications,locked'}
    api = 'security/accounts/%s/%s' % (owner_uuid, name)
    response, error = self.rest_api.get(api, query)
    if error:
        self.module.fail_json(msg='Error while fetching user details: %s' % error)
    if response:
        for application in response['applications']:
            if application.get('second_authentication_method') == 'none':
                application['second_authentication_method'] = None
            application.pop('is_ldap_fastbind', None)
        return_value = {'role_name': response['role']['name'], 'applications': response['applications']}
        if 'locked' in response:
            return_value['lock_user'] = response['locked']
    return return_value