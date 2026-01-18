from __future__ import absolute_import, division, print_function
import hmac
import itertools
import re
import traceback
from base64 import b64decode
from hashlib import md5, sha256
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils import \
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def has_table_privileges(cursor, user, table, privs):
    """
    Return the difference between the privileges that a user already has and
    the privileges that they desire to have.

    :returns: tuple of:
        * privileges that they have and were requested
        * privileges they currently hold but were not requested
        * privileges requested that they do not hold
    """
    cur_privs = get_table_privileges(cursor, user, table)
    have_currently = cur_privs.intersection(privs)
    other_current = cur_privs.difference(privs)
    desired = privs.difference(cur_privs)
    return (have_currently, other_current, desired)