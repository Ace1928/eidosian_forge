from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
def role_exists(self):
    """Return a query to check if a role with self.name exists in a database.

        Returns:
            tuple: (query_string, tuple_containing_parameters).
        """
    return ("SELECT count(*) FROM mysql.user WHERE user = %s AND is_role  = 'Y'", (self.name,))