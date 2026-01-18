from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def set_lang_owner(cursor, lang, owner):
    """Set language owner.

    Args:
        cursor (cursor): psycopg cursor object.
        lang (str): language name.
        owner (str): name of new owner.
    """
    query = 'ALTER LANGUAGE "%s" OWNER TO "%s"' % (lang, owner)
    executed_queries.append(query)
    cursor.execute(query)
    return True