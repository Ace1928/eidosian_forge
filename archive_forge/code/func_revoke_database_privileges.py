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
def revoke_database_privileges(cursor, user, db, privs):
    privs = ', '.join(privs)
    if user == 'PUBLIC':
        query = 'REVOKE %s ON DATABASE %s FROM PUBLIC' % (privs, pg_quote_identifier(db, 'database'))
    else:
        query = 'REVOKE %s ON DATABASE %s FROM "%s"' % (privs, pg_quote_identifier(db, 'database'), user)
    executed_queries.append(query)
    cursor.execute(query)