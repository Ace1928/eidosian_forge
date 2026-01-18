from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def get_user_entry(self, uid):
    """
            get_user_entry returns an LDAP group entry for the given user UID by searching the internal cache
            of the LDAPInterface first, then sending an LDAP query if the cache did not contain the entry.
        """
    if uid in self.cached_users:
        return (self.cached_users.get(uid), None)
    entry, err = self.userQuery.ldap_search(self.connection, uid, self.required_user_attributes)
    if err:
        return (None, err)
    self.cached_users[uid] = entry
    return (entry, None)