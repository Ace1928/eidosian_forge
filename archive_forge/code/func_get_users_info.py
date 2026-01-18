from __future__ import absolute_import, division, print_function
from uuid import UUID
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def get_users_info(self, dbname):
    """Gather information about users.

        Args:
            dbname (str): Database name to get user info from.

        Returns a dictionary with user information for the given db.
        """
    db = self.client[dbname]
    result = db.command({'usersInfo': 1})['users']
    users_dict = {}
    for elem in result:
        users_dict[elem['user']] = {}
        for key, val in iteritems(elem):
            if key in ['user', 'db']:
                continue
            if isinstance(val, UUID):
                val = val.hex
            users_dict[elem['user']][key] = val
    return {dbname: users_dict}