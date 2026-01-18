from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def lang_exists(cursor, lang):
    """Checks if language exists for db"""
    query = 'SELECT lanname FROM pg_language WHERE lanname = %(lang)s'
    cursor.execute(query, {'lang': lang})
    return cursor.rowcount > 0