from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_role_rest(self, modify):
    privileges = self.get_role_privileges_rest()
    modify_privilege = []
    for privilege in modify['privileges']:
        path = privilege['path']
        modify_privilege.append(path)
        if path not in privileges:
            self.create_role_privilege(privilege)
        elif privilege.get('query'):
            if not privileges[path].get('query'):
                self.modify_role_privilege(privilege, path)
            elif privilege['query'] != privileges[path]['query']:
                self.modify_role_privilege(privilege, path)
        elif privilege.get('access') and privilege['access'] != privileges[path]['access']:
            self.modify_role_privilege(privilege, path)
    for privilege_path in privileges:
        if privilege_path not in modify_privilege:
            self.delete_role_privilege(privilege_path)