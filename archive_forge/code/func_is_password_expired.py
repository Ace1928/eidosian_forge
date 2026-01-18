from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def is_password_expired(cursor, user, host):
    """Function to check if password is expired

    Args:
        cursor (cursor): DB driver cursor object.
        user (str): User name.
        host (str): User hostname.

    Returns:
        expired (bool): True if expired, else False.
    """
    statement = ('SELECT password_expired FROM mysql.user             WHERE User = %s AND Host = %s', (user, host))
    cursor.execute(*statement)
    expired = cursor.fetchone()[0]
    if str(expired) == 'Y':
        return True
    return False