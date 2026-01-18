from __future__ import absolute_import, division, print_function
import re
from fnmatch import fnmatch
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def get_pub_info(self):
    """Get publication statistics."""
    query = 'SELECT p.*, r.rolname AS ownername FROM pg_catalog.pg_publication AS p JOIN pg_catalog.pg_roles AS r ON p.pubowner = r.oid'
    result = self.__exec_sql(query)
    if result:
        result = [dict(row) for row in result]
    else:
        return {}
    publications = {}
    for elem in result:
        if not publications.get(elem['pubname']):
            publications[elem['pubname']] = {}
        for key, val in iteritems(elem):
            if key != 'pubname':
                publications[elem['pubname']][key] = val
    return publications