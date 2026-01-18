from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
def set_admin(self, admin):
    """Set an admin of a role.

        TODO: Implement changing when ALTER ROLE statement to
            change role's admin gets supported.

        Args:
            admin (tuple): Admin user of the role in the form (username, hostname).
        """
    admin_user = admin[0]
    admin_host = admin[1]
    current_admin = self.get_admin()
    if (admin_user, admin_host) != current_admin:
        msg = 'The "admin" option value and the current roles admin (%s@%s) don not match. Ignored. To change the admin, you need to drop and create the role again.' % (current_admin[0], current_admin[1])
        self.module.warn(msg)