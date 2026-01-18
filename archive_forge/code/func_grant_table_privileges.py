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
def grant_table_privileges(cursor, user, table, privs):
    privs = ', '.join(privs)
    query = 'GRANT %s ON TABLE %s TO "%s"' % (privs, pg_quote_identifier(table, 'table'), user)
    executed_queries.append(query)
    cursor.execute(query)