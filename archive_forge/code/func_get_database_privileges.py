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
def get_database_privileges(cursor, user, db):
    priv_map = {'C': 'CREATE', 'T': 'TEMPORARY', 'c': 'CONNECT'}
    query = 'SELECT datacl::text FROM pg_database WHERE datname = %s'
    cursor.execute(query, (db,))
    datacl = cursor.fetchone()['datacl']
    if datacl is None:
        return set()
    r = re.search('%s\\\\?"?=(C?T?c?)/[^,]+,?' % user, datacl)
    if r is None:
        return set()
    o = set()
    for v in r.group(1):
        o.add(priv_map[v])
    return normalize_privileges(o, 'database')