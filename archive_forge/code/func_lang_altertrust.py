from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def lang_altertrust(cursor, lang, trust):
    """Changes if language is trusted for db"""
    query = 'UPDATE pg_language SET lanpltrusted = %(trust)s WHERE lanname = %(lang)s'
    cursor.execute(query, {'trust': trust, 'lang': lang})
    executed_queries.append(cursor.mogrify(query, {'trust': trust, 'lang': lang}))
    return True