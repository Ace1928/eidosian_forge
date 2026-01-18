from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def attributes_get(cursor, user, host):
    """Get attributes for a given user.

    Args:
        cursor (cursor): DB driver cursor object.
        user (str): User name.
        host (str): User host name.

    Returns:
        None if the user does not exist or the user has no attributes set, otherwise a dict of attributes set on the user
    """
    cursor.execute('SELECT attribute FROM INFORMATION_SCHEMA.USER_ATTRIBUTES WHERE user = %s AND host = %s', (user, host))
    r = cursor.fetchone()
    j = json.loads(r[0]) if r and r[0] else None
    return j if j else None