from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def check_role_exists(self, role, fail_on_role=True):
    """Check the role exists or not.

        Arguments:
            role (str): Role name.
            fail_on_role (bool): If True, fail when the role does not exist.
                Otherwise just warn and continue.
        """
    if not self.__role_exists(role):
        if fail_on_role:
            self.module.fail_json(msg="Role '%s' does not exist" % role)
        else:
            self.module.warn("Role '%s' does not exist, pass" % role)
        return False
    else:
        return True