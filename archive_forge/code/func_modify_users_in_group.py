from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_users_in_group(self):
    """
        Add/delete one or many users in a UNIX group

        :return: None
        """
    current_users = self.get_unix_group().get('users')
    expect_users = self.parameters.get('users')
    if current_users is None:
        current_users = []
    if expect_users[0] == '' and len(expect_users) == 1:
        expect_users = []
    users_to_remove = list(set(current_users) - set(expect_users))
    users_to_add = list(set(expect_users) - set(current_users))
    if len(users_to_add) > 0:
        for user in users_to_add:
            add_user = netapp_utils.zapi.NaElement('name-mapping-unix-group-add-user')
            group_details = {'group-name': self.parameters['name'], 'user-name': user}
            add_user.translate_struct(group_details)
            try:
                self.server.invoke_successfully(add_user, enable_tunneling=True)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error adding user %s to UNIX group %s: %s' % (user, self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    if len(users_to_remove) > 0:
        for user in users_to_remove:
            delete_user = netapp_utils.zapi.NaElement('name-mapping-unix-group-delete-user')
            group_details = {'group-name': self.parameters['name'], 'user-name': user}
            delete_user.translate_struct(group_details)
            try:
                self.server.invoke_successfully(delete_user, enable_tunneling=True)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error deleting user %s from UNIX group %s: %s' % (user, self.parameters['name'], to_native(error)), exception=traceback.format_exc())