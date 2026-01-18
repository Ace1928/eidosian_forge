from __future__ import absolute_import, division, print_function
import os
import subprocess
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def db_matches(cursor, db, owner, template, encoding, lc_collate, lc_ctype, icu_locale, locale_provider, conn_limit, tablespace, comment):
    if not db_exists(cursor, db):
        return False
    else:
        db_info = get_db_info(cursor, db)
        if db_info['comment'] is None:
            db_info['comment'] = ''
        if encoding and get_encoding_id(cursor, encoding) != db_info['encoding_id']:
            return False
        elif lc_collate and lc_collate != db_info['lc_collate']:
            return False
        elif lc_ctype and lc_ctype != db_info['lc_ctype']:
            return False
        elif icu_locale and icu_locale != db_info['icu_locale']:
            return False
        elif locale_provider and locale_provider != db_info['locale_provider']:
            return False
        elif owner and owner != db_info['owner']:
            return False
        elif conn_limit and conn_limit != str(db_info['conn_limit']):
            return False
        elif tablespace and tablespace != db_info['tablespace']:
            return False
        elif comment is not None and comment != db_info['comment']:
            return False
        else:
            return True