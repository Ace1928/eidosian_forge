from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
def check_users_in_db(self, users):
    """Check if users exist in a database.

        Args:
            users (list): List of tuples (username, hostname) to check.
        """
    for user in users:
        if user not in self.users:
            msg = 'User / role `%s` with host `%s` does not exist' % (user[0], user[1])
            self.module.fail_json(msg=msg)